# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from typing import Dict, Iterable, List, Union

from openai.types.chat import (
    ChatCompletionAssistantMessageParam as OpenAIChatCompletionAssistantMessage,
)
from openai.types.chat import (
    ChatCompletionContentPartImageParam as OpenAIChatCompletionContentPartImageParam,
)
from openai.types.chat import (
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam,
)
from openai.types.chat import (
    ChatCompletionContentPartTextParam as OpenAIChatCompletionContentPartTextParam,
)
from openai.types.chat import (
    ChatCompletionMessageParam as OpenAIChatCompletionMessage,
)
from openai.types.chat import (
    ChatCompletionMessageToolCallParam as OpenAIChatCompletionMessageToolCall,
)
from openai.types.chat import (
    ChatCompletionSystemMessageParam as OpenAIChatCompletionSystemMessage,
)
from openai.types.chat import (
    ChatCompletionToolMessageParam as OpenAIChatCompletionToolMessage,
)
from openai.types.chat import (
    ChatCompletionUserMessageParam as OpenAIChatCompletionUserMessage,
)
from openai.types.chat.chat_completion_content_part_image_param import (
    ImageURL as OpenAIImageURL,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    Function as OpenAIFunction,
)

from llama_stack.apis.common.content_types import (
    ImageContentItem,
    InterleavedContent,
    TextContentItem,
)
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    CompletionMessage,
    JsonSchemaResponseFormat,
    Message,
    SystemMessage,
    ToolChoice,
    ToolResponseMessage,
    UserMessage,
)
from llama_stack.models.llama.datatypes import BuiltinTool
from llama_stack.providers.remote.inference.sambanova.config import SambaNovaImplConfig
from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin
from llama_stack.providers.utils.inference.openai_compat import (
    convert_tooldef_to_openai_tool,
    get_sampling_options,
)
from llama_stack.providers.utils.inference.prompt_adapter import convert_image_content_to_url

from .models import MODEL_ENTRIES


async def convert_message_to_openai_dict_with_b64_images(
    message: Message | Dict,
) -> OpenAIChatCompletionMessage:
    """
    Convert a Message to an OpenAI API-compatible dictionary.
    """
    # users can supply a dict instead of a Message object, we'll
    # convert it to a Message object and proceed with some type safety.
    if isinstance(message, dict):
        if "role" not in message:
            raise ValueError("role is required in message")
        if message["role"] == "user":
            message = UserMessage(**message)
        elif message["role"] == "assistant":
            message = CompletionMessage(**message)
        elif message["role"] == "tool":
            message = ToolResponseMessage(**message)
        elif message["role"] == "system":
            message = SystemMessage(**message)
        else:
            raise ValueError(f"Unsupported message role: {message['role']}")

    # Map Llama Stack spec to OpenAI spec -
    #  str -> str
    #  {"type": "text", "text": ...} -> {"type": "text", "text": ...}
    #  {"type": "image", "image": {"url": {"uri": ...}}} -> {"type": "image_url", "image_url": {"url": ...}}
    #  {"type": "image", "image": {"data": ...}} -> {"type": "image_url", "image_url": {"url": "data:image/?;base64,..."}}
    #  List[...] -> List[...]
    async def _convert_message_content(
        content: InterleavedContent,
    ) -> Union[str, Iterable[OpenAIChatCompletionContentPartParam]]:
        async def impl(
            content_: InterleavedContent,
        ) -> Union[str, OpenAIChatCompletionContentPartParam, List[OpenAIChatCompletionContentPartParam]]:
            # Llama Stack and OpenAI spec match for str and text input
            if isinstance(content_, str):
                return content_
            elif isinstance(content_, TextContentItem):
                return OpenAIChatCompletionContentPartTextParam(
                    type="text",
                    text=content_.text,
                )
            elif isinstance(content_, ImageContentItem):
                return OpenAIChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url=OpenAIImageURL(url=await convert_image_content_to_url(content_, download=True)),
                )
            elif isinstance(content_, list):
                return [await impl(item) for item in content_]
            else:
                raise ValueError(f"Unsupported content type: {type(content_)}")

        ret = await impl(content)

        # OpenAI*Message expects a str or list
        if isinstance(ret, str) or isinstance(ret, list):
            return ret
        else:
            return [ret]

    out: OpenAIChatCompletionMessage = None
    if isinstance(message, UserMessage):
        out = OpenAIChatCompletionUserMessage(
            role="user",
            content=await _convert_message_content(message.content),
        )
    elif isinstance(message, CompletionMessage):
        out = OpenAIChatCompletionAssistantMessage(
            role="assistant",
            content=await _convert_message_content(message.content),
            tool_calls=[
                OpenAIChatCompletionMessageToolCall(
                    id=tool.call_id,
                    function=OpenAIFunction(
                        name=tool.tool_name if not isinstance(tool.tool_name, BuiltinTool) else tool.tool_name.value,
                        arguments=json.dumps(tool.arguments),
                    ),
                    type="function",
                )
                for tool in message.tool_calls
            ]
            or None,
        )
    elif isinstance(message, ToolResponseMessage):
        out = OpenAIChatCompletionToolMessage(
            role="tool",
            tool_call_id=message.call_id,
            content=await _convert_message_content(message.content),
        )
    elif isinstance(message, SystemMessage):
        out = OpenAIChatCompletionSystemMessage(
            role="system",
            content=await _convert_message_content(message.content),
        )
    else:
        raise ValueError(f"Unsupported message type: {type(message)}")

    return out


class SambaNovaInferenceAdapter(LiteLLMOpenAIMixin):
    _config: SambaNovaImplConfig

    def __init__(self, config: SambaNovaImplConfig):
        LiteLLMOpenAIMixin.__init__(
            self,
            model_entries=MODEL_ENTRIES,
            api_key_from_config=config.api_key,
            provider_data_api_key_field="sambanova_api_key",
        )
        self.config = config

    async def _get_params(self, request: ChatCompletionRequest) -> dict:
        input_dict = {}

        input_dict["messages"] = [await convert_message_to_openai_dict_with_b64_images(m) for m in request.messages]
        if fmt := request.response_format:
            if not isinstance(fmt, JsonSchemaResponseFormat):
                raise ValueError(
                    f"Unsupported response format: {type(fmt)}. Only JsonSchemaResponseFormat is supported."
                )

            fmt = fmt.json_schema
            name = fmt["title"]
            del fmt["title"]
            fmt["additionalProperties"] = False

            # Apply additionalProperties: False recursively to all objects
            fmt = self._add_additional_properties_recursive(fmt)

            input_dict["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "schema": fmt,
                    "strict": True,
                },
            }
        if request.tools:
            input_dict["tools"] = [convert_tooldef_to_openai_tool(tool) for tool in request.tools]
            if request.tool_config.tool_choice:
                input_dict["tool_choice"] = (
                    request.tool_config.tool_choice.value
                    if isinstance(request.tool_config.tool_choice, ToolChoice)
                    else request.tool_config.tool_choice
                )

        provider_data = self.get_request_provider_data()
        key_field = self.provider_data_api_key_field
        if provider_data and getattr(provider_data, key_field, None):
            api_key = getattr(provider_data, key_field)
        else:
            api_key = self.api_key_from_config

        return {
            "model": request.model,
            "api_key": api_key,
            "api_base": self.config.url,
            **input_dict,
            "stream": request.stream,
            **get_sampling_options(request.sampling_params),
        }

    async def initialize(self):
        await super().initialize()

    async def shutdown(self):
        await super().shutdown()
