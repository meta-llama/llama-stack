# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid

from llama_stack.apis.agents.openai_responses import (
    OpenAIResponseInputTool,
    OpenAIResponseMessage,
    OpenAIResponseOutputMessageContentOutputText,
)
from llama_stack.apis.inference import (
    OpenAIChatCompletionContentPartTextParam,
    OpenAIChatCompletionToolCall,
    OpenAIChoice,
)


async def convert_chat_choice_to_response_message(choice: OpenAIChoice) -> OpenAIResponseMessage:
    """Convert an OpenAI Chat Completion choice into an OpenAI Response output message."""
    output_content = ""
    if isinstance(choice.message.content, str):
        output_content = choice.message.content
    elif isinstance(choice.message.content, OpenAIChatCompletionContentPartTextParam):
        output_content = choice.message.content.text
    else:
        raise ValueError(
            f"Llama Stack OpenAI Responses does not yet support output content type: {type(choice.message.content)}"
        )

    return OpenAIResponseMessage(
        id=f"msg_{uuid.uuid4()}",
        content=[OpenAIResponseOutputMessageContentOutputText(text=output_content)],
        status="completed",
        role="assistant",
    )


def is_function_tool_call(
    tool_call: OpenAIChatCompletionToolCall,
    tools: list[OpenAIResponseInputTool],
) -> bool:
    if not tool_call.function:
        return False
    for t in tools:
        if t.type == "function" and t.name == tool_call.function.name:
            return True
    return False
