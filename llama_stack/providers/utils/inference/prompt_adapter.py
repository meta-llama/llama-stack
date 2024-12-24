# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import base64
import io
import json
import logging
import re
from typing import List, Optional, Tuple, Union

import httpx
from llama_models.datatypes import is_multimodal, ModelFamily
from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.datatypes import (
    RawContent,
    RawContentItem,
    RawMediaItem,
    RawMessage,
    RawTextItem,
    Role,
    ToolPromptFormat,
)
from llama_models.llama3.prompt_templates import (
    BuiltinToolGenerator,
    FunctionTagCustomToolGenerator,
    JsonCustomToolGenerator,
    PythonListCustomToolGenerator,
    SystemDefaultGenerator,
)
from llama_models.sku_list import resolve_model
from PIL import Image as PIL_Image

from llama_stack.apis.common.content_types import (
    ImageContentItem,
    InterleavedContent,
    InterleavedContentItem,
    TextContentItem,
    URL,
)
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    CompletionRequest,
    Message,
    ResponseFormat,
    ResponseFormatType,
    SystemMessage,
    ToolChoice,
    UserMessage,
)
from llama_stack.providers.utils.inference import supported_inference_models

log = logging.getLogger(__name__)


class ChatCompletionRequestWithRawContent(ChatCompletionRequest):
    messages: List[RawMessage]


class CompletionRequestWithRawContent(CompletionRequest):
    content: RawContent


def interleaved_content_as_str(content: InterleavedContent, sep: str = " ") -> str:
    def _process(c) -> str:
        if isinstance(c, str):
            return c
        elif isinstance(c, ImageContentItem):
            return "<image>"
        elif isinstance(c, TextContentItem):
            return c.text
        else:
            raise ValueError(f"Unsupported content type: {type(c)}")

    if isinstance(content, list):
        return sep.join(_process(c) for c in content)
    else:
        return _process(content)


async def convert_request_to_raw(
    request: Union[ChatCompletionRequest, CompletionRequest],
) -> Union[ChatCompletionRequestWithRawContent, CompletionRequestWithRawContent]:
    if isinstance(request, ChatCompletionRequest):
        messages = []
        for m in request.messages:
            content = await interleaved_content_convert_to_raw(m.content)
            d = m.model_dump()
            d["content"] = content
            messages.append(RawMessage(**d))

        d = request.model_dump()
        d["messages"] = messages
        request = ChatCompletionRequestWithRawContent(**d)
    else:
        d = request.model_dump()
        d["content"] = await interleaved_content_convert_to_raw(request.content)
        request = CompletionRequestWithRawContent(**d)

    return request


async def interleaved_content_convert_to_raw(
    content: InterleavedContent,
) -> RawContent:
    """Download content from URLs / files etc. so plain bytes can be sent to the model"""

    async def _localize_single(c: str | InterleavedContentItem) -> str | RawContentItem:
        if isinstance(c, str):
            return RawTextItem(text=c)
        elif isinstance(c, TextContentItem):
            return RawTextItem(text=c.text)
        elif isinstance(c, ImageContentItem):
            # load image and return PIL version
            img = c.data
            if isinstance(img, URL):
                if img.uri.startswith("data"):
                    match = re.match(r"data:image/(\w+);base64,(.+)", img.uri)
                    if not match:
                        raise ValueError("Invalid data URL format")
                    _, image_data = match.groups()
                    data = base64.b64decode(image_data)
                elif img.uri.startswith("file://"):
                    path = img.uri[len("file://") :]
                    with open(path, "rb") as f:
                        data = f.read()  # type: ignore
                elif img.uri.startswith("http"):
                    async with httpx.AsyncClient() as client:
                        response = await client.get(img.uri)
                        data = response.content
                else:
                    raise ValueError("Unsupported URL type")
            else:
                data = c.data
            return RawMediaItem(data=data)
        else:
            raise ValueError(f"Unsupported content type: {type(c)}")

    if isinstance(content, list):
        return await asyncio.gather(*(_localize_single(c) for c in content))
    else:
        return await _localize_single(content)


def content_has_media(content: InterleavedContent):
    def _has_media_content(c):
        return isinstance(c, ImageContentItem)

    if isinstance(content, list):
        return any(_has_media_content(c) for c in content)
    else:
        return _has_media_content(content)


def messages_have_media(messages: List[Message]):
    return any(content_has_media(m.content) for m in messages)


def request_has_media(request: Union[ChatCompletionRequest, CompletionRequest]):
    if isinstance(request, ChatCompletionRequest):
        return messages_have_media(request.messages)
    else:
        return content_has_media(request.content)


async def localize_image_content(media: ImageContentItem) -> Tuple[bytes, str]:
    if media.url and media.url.uri.startswith("http"):
        async with httpx.AsyncClient() as client:
            r = await client.get(media.url.uri)
            content = r.content
            content_type = r.headers.get("content-type")
            if content_type:
                format = content_type.split("/")[-1]
            else:
                format = "png"
        return content, format
    else:
        image = PIL_Image.open(io.BytesIO(media.data))
        return media.data, image.format


async def convert_image_content_to_url(
    media: ImageContentItem, download: bool = False, include_format: bool = True
) -> str:
    if media.url and not download:
        return media.url.uri

    content, format = await localize_image_content(media)
    if include_format:
        return f"data:image/{format};base64," + base64.b64encode(content).decode(
            "utf-8"
        )
    else:
        return base64.b64encode(content).decode("utf-8")


async def completion_request_to_prompt(
    request: CompletionRequest, formatter: ChatFormat
) -> str:
    content = augment_content_with_response_format_prompt(
        request.response_format, request.content
    )
    request.content = content
    request = await convert_request_to_raw(request)
    model_input = formatter.encode_content(request.content)
    return formatter.tokenizer.decode(model_input.tokens)


async def completion_request_to_prompt_model_input_info(
    request: CompletionRequest, formatter: ChatFormat
) -> Tuple[str, int]:
    content = augment_content_with_response_format_prompt(
        request.response_format, request.content
    )
    request.content = content
    request = await convert_request_to_raw(request)
    model_input = formatter.encode_content(request.content)
    return (formatter.tokenizer.decode(model_input.tokens), len(model_input.tokens))


def augment_content_with_response_format_prompt(response_format, content):
    if fmt_prompt := response_format_prompt(response_format):
        if isinstance(content, list):
            return content + [fmt_prompt]
        else:
            return [content, fmt_prompt]

    return content


async def chat_completion_request_to_prompt(
    request: ChatCompletionRequest, llama_model: str, formatter: ChatFormat
) -> str:
    messages = chat_completion_request_to_messages(request, llama_model)
    request.messages = messages
    request = await convert_request_to_raw(request)
    model_input = formatter.encode_dialog_prompt(request.messages)
    return formatter.tokenizer.decode(model_input.tokens)


async def chat_completion_request_to_model_input_info(
    request: ChatCompletionRequest, llama_model: str, formatter: ChatFormat
) -> Tuple[str, int]:
    messages = chat_completion_request_to_messages(request, llama_model)
    request.messages = messages
    request = await convert_request_to_raw(request)
    model_input = formatter.encode_dialog_prompt(request.messages)
    return (
        formatter.tokenizer.decode(model_input.tokens),
        len(model_input.tokens),
    )


def chat_completion_request_to_messages(
    request: ChatCompletionRequest,
    llama_model: str,
) -> List[Message]:
    """Reads chat completion request and augments the messages to handle tools.
    For eg. for llama_3_1, add system message with the appropriate tools or
    add user messsage for custom tools, etc.
    """
    model = resolve_model(llama_model)
    if model is None:
        log.error(f"Could not resolve model {llama_model}")
        return request.messages

    allowed_models = supported_inference_models()
    descriptors = [m.descriptor() for m in allowed_models]
    if model.descriptor() not in descriptors:
        log.error(f"Unsupported inference model? {model.descriptor()}")
        return request.messages

    if model.model_family == ModelFamily.llama3_1 or (
        model.model_family == ModelFamily.llama3_2
        and is_multimodal(model.core_model_id)
    ):
        # llama3.1 and llama3.2 multimodal models follow the same tool prompt format
        messages = augment_messages_for_tools_llama_3_1(request)
    elif model.model_family in (ModelFamily.llama3_2, ModelFamily.llama3_3):
        # llama3.2 and llama3.3 models follow the same tool prompt format
        messages = augment_messages_for_tools_llama_3_2(request)
    else:
        messages = request.messages

    if fmt_prompt := response_format_prompt(request.response_format):
        messages.append(UserMessage(content=fmt_prompt))

    return messages


def response_format_prompt(fmt: Optional[ResponseFormat]):
    if not fmt:
        return None

    if fmt.type == ResponseFormatType.json_schema.value:
        return f"Please respond in JSON format with the schema: {json.dumps(fmt.json_schema)}"
    elif fmt.type == ResponseFormatType.grammar.value:
        raise NotImplementedError("Grammar response format not supported yet")
    else:
        raise ValueError(f"Unknown response format {fmt.type}")


def augment_messages_for_tools_llama_3_1(
    request: ChatCompletionRequest,
) -> List[Message]:
    assert request.tool_choice == ToolChoice.auto, "Only `ToolChoice.auto` supported"

    existing_messages = request.messages
    existing_system_message = None
    if existing_messages[0].role == Role.system.value:
        existing_system_message = existing_messages.pop(0)

    assert (
        existing_messages[0].role != Role.system.value
    ), "Should only have 1 system message"

    messages = []

    default_gen = SystemDefaultGenerator()
    default_template = default_gen.gen()

    sys_content = ""

    tool_template = None
    if request.tools:
        tool_gen = BuiltinToolGenerator()
        tool_template = tool_gen.gen(request.tools)

        sys_content += tool_template.render()
        sys_content += "\n"

    sys_content += default_template.render()

    if existing_system_message:
        # TODO: this fn is needed in many places
        def _process(c):
            if isinstance(c, str):
                return c
            else:
                return "<media>"

        sys_content += "\n"

        if isinstance(existing_system_message.content, str):
            sys_content += _process(existing_system_message.content)
        elif isinstance(existing_system_message.content, list):
            sys_content += "\n".join(
                [_process(c) for c in existing_system_message.content]
            )

    messages.append(SystemMessage(content=sys_content))

    has_custom_tools = any(isinstance(dfn.tool_name, str) for dfn in request.tools)
    if has_custom_tools:
        if request.tool_prompt_format == ToolPromptFormat.json:
            tool_gen = JsonCustomToolGenerator()
        elif request.tool_prompt_format == ToolPromptFormat.function_tag:
            tool_gen = FunctionTagCustomToolGenerator()
        else:
            raise ValueError(
                f"Non supported ToolPromptFormat {request.tool_prompt_format}"
            )

        custom_tools = [t for t in request.tools if isinstance(t.tool_name, str)]
        custom_template = tool_gen.gen(custom_tools)
        messages.append(UserMessage(content=custom_template.render()))

    # Add back existing messages from the request
    messages += existing_messages

    return messages


def augment_messages_for_tools_llama_3_2(
    request: ChatCompletionRequest,
) -> List[Message]:
    assert request.tool_choice == ToolChoice.auto, "Only `ToolChoice.auto` supported"

    existing_messages = request.messages
    existing_system_message = None
    if existing_messages[0].role == Role.system.value:
        existing_system_message = existing_messages.pop(0)

    assert (
        existing_messages[0].role != Role.system.value
    ), "Should only have 1 system message"

    messages = []
    sys_content = ""
    custom_tools, builtin_tools = [], []
    for t in request.tools:
        if isinstance(t.tool_name, str):
            custom_tools.append(t)
        else:
            builtin_tools.append(t)

    tool_template = None
    if builtin_tools:
        tool_gen = BuiltinToolGenerator()
        tool_template = tool_gen.gen(builtin_tools)

        sys_content += tool_template.render()
        sys_content += "\n"

    custom_tools = [dfn for dfn in request.tools if isinstance(dfn.tool_name, str)]
    if custom_tools:
        if request.tool_prompt_format != ToolPromptFormat.python_list:
            raise ValueError(
                f"Non supported ToolPromptFormat {request.tool_prompt_format}"
            )

        tool_gen = PythonListCustomToolGenerator()
        tool_template = tool_gen.gen(custom_tools)

        sys_content += tool_template.render()
        sys_content += "\n"

    if existing_system_message:
        sys_content += interleaved_content_as_str(
            existing_system_message.content, sep="\n"
        )

    messages.append(SystemMessage(content=sys_content))

    # Add back existing messages from the request
    messages += existing_messages
    return messages
