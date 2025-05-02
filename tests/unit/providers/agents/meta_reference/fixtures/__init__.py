# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import yaml

from llama_stack.apis.inference.inference import (
    OpenAIAssistantMessageParam,
    OpenAIChatCompletion,
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolCallFunction,
    OpenAIChoice,
)


def load_chat_completion_fixture(filename: str) -> OpenAIChatCompletion:
    """
    Load a YAML fixture file and convert it to an OpenAIChatCompletion object.

    Args:
        filename: Name of the YAML file (without path)

    Returns:
        OpenAIChatCompletion object
    """
    fixtures_dir = os.path.dirname(os.path.abspath(__file__))
    fixture_path = os.path.join(fixtures_dir, filename)

    with open(fixture_path) as f:
        data = yaml.safe_load(f)

    choices = []
    for choice_data in data.get("choices", []):
        message_data = choice_data.get("message", {})

        # Handle tool calls if present
        tool_calls = None
        if "tool_calls" in message_data:
            tool_calls = []
            for tool_call_data in message_data.get("tool_calls", []):
                function_data = tool_call_data.get("function", {})
                function = OpenAIChatCompletionToolCallFunction(
                    name=function_data.get("name"),
                    arguments=function_data.get("arguments"),
                )
                tool_call = OpenAIChatCompletionToolCall(
                    id=tool_call_data.get("id"),
                    type=tool_call_data.get("type"),
                    function=function,
                )
                tool_calls.append(tool_call)

        message = OpenAIAssistantMessageParam(
            content=message_data.get("content"),
            tool_calls=tool_calls,
        )

        choice = OpenAIChoice(
            message=message,
            finish_reason=choice_data.get("finish_reason"),
            index=choice_data.get("index", 0),
        )
        choices.append(choice)

    return OpenAIChatCompletion(
        id=data.get("id"),
        choices=choices,
        created=data.get("created"),
        model=data.get("model"),
    )
