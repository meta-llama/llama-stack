# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import io
import json
import logging
from typing import Tuple

import httpx

from llama_models.llama3.api.chat_format import ChatFormat
from PIL import Image as PIL_Image
from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.inference import *  # noqa: F403
from llama_models.datatypes import ModelFamily
from llama_models.llama3.prompt_templates import (
    BuiltinToolGenerator,
    FunctionTagCustomToolGenerator,
    JsonCustomToolGenerator,
    PythonListCustomToolGenerator,
    SystemDefaultGenerator,
)
from llama_models.sku_list import resolve_model

from llama_stack.providers.utils.inference import supported_inference_models

log = logging.getLogger(__name__)


def content_has_media(content: InterleavedTextMedia):
    def _has_media_content(c):
        return isinstance(c, ImageMedia)

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


async def convert_image_media_to_url(
    media: ImageMedia, download: bool = False, include_format: bool = True
) -> str:
    if isinstance(media.image, PIL_Image.Image):
        if media.image.format == "PNG":
            format = "png"
        elif media.image.format == "GIF":
            format = "gif"
        elif media.image.format == "JPEG":
            format = "jpeg"
        else:
            raise ValueError(f"Unsupported image format {media.image.format}")

        bytestream = io.BytesIO()
        media.image.save(bytestream, format=media.image.format)
        bytestream.seek(0)
        content = bytestream.getvalue()
    else:
        if not download:
            return media.image.uri
        else:
            assert isinstance(media.image, URL)
            async with httpx.AsyncClient() as client:
                r = await client.get(media.image.uri)
                content = r.content
                content_type = r.headers.get("content-type")
                if content_type:
                    format = content_type.split("/")[-1]
                else:
                    format = "png"

    if include_format:
        return f"data:image/{format};base64," + base64.b64encode(content).decode(
            "utf-8"
        )
    else:
        return base64.b64encode(content).decode("utf-8")


# TODO: name this function better! this is about OpenAI compatibile image
# media conversion of the message. this should probably go in openai_compat.py
async def convert_message_to_dict(message: Message, download: bool = False) -> dict:
    async def _convert_content(content) -> dict:
        if isinstance(content, ImageMedia):
            return {
                "type": "image_url",
                "image_url": {
                    "url": await convert_image_media_to_url(content, download=download),
                },
            }
        else:
            assert isinstance(content, str)
            return {"type": "text", "text": content}

    if isinstance(message.content, list):
        content = [await _convert_content(c) for c in message.content]
    else:
        content = [await _convert_content(message.content)]

    return {
        "role": message.role,
        "content": content,
    }


def completion_request_to_prompt(
    request: CompletionRequest, formatter: ChatFormat
) -> str:
    content = augment_content_with_response_format_prompt(
        request.response_format, request.content
    )
    model_input = formatter.encode_content(content)
    return formatter.tokenizer.decode(model_input.tokens)


def completion_request_to_prompt_model_input_info(
    request: CompletionRequest, formatter: ChatFormat
) -> Tuple[str, int]:
    content = augment_content_with_response_format_prompt(
        request.response_format, request.content
    )
    model_input = formatter.encode_content(content)
    return (formatter.tokenizer.decode(model_input.tokens), len(model_input.tokens))


def augment_content_with_response_format_prompt(response_format, content):
    if fmt_prompt := response_format_prompt(response_format):
        if isinstance(content, list):
            return content + [fmt_prompt]
        else:
            return [content, fmt_prompt]

    return content


def chat_completion_request_to_prompt(
    request: ChatCompletionRequest, llama_model: str, formatter: ChatFormat
) -> str:
    messages = chat_completion_request_to_messages(request, llama_model)
    model_input = formatter.encode_dialog_prompt(messages)
    return formatter.tokenizer.decode(model_input.tokens)


def chat_completion_request_to_model_input_info(
    request: ChatCompletionRequest, llama_model: str, formatter: ChatFormat
) -> Tuple[str, int]:
    messages = chat_completion_request_to_messages(request, llama_model)
    model_input = formatter.encode_dialog_prompt(messages)
    return (
        formatter.tokenizer.decode(model_input.tokens),
        len(model_input.tokens),
    )


def chat_completion_request_to_messages(
    request: ChatCompletionRequest,
    llama_model: str,
    return_dict: bool = False,
) -> Union[List[Message], List[Dict[str, str]]]:
    """Reads chat completion request and augments the messages to handle tools.
    For eg. for llama_3_1, add system message with the appropriate tools or
    add user messsage for custom tools, etc.
    If return_dict is set, returns a list of the messages dictionaries instead of objects.
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
    elif model.model_family == ModelFamily.llama3_2:
        messages = augment_messages_for_tools_llama_3_2(request)
    else:
        messages = request.messages

    if fmt_prompt := response_format_prompt(request.response_format):
        messages.append(UserMessage(content=fmt_prompt))

    if return_dict:
        return [{'role': m.role, 'content': m.content} for m in messages]
    else:
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
        sys_content += interleaved_text_media_as_str(
            existing_system_message.content, sep="\n"
        )

    messages.append(SystemMessage(content=sys_content))

    # Add back existing messages from the request
    messages += existing_messages
    return messages
