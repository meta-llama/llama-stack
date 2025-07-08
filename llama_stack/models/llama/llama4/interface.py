# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from pathlib import Path

from termcolor import colored

from ..datatypes import (
    BuiltinTool,
    RawMessage,
    StopReason,
    ToolCall,
    ToolDefinition,
    ToolPromptFormat,
)
from ..llama3.prompt_templates import (
    BuiltinToolGenerator,
    ToolResponseGenerator,
)
from .chat_format import ChatFormat
from .prompt_templates.system_prompts import PythonListCustomToolGenerator
from .tokenizer import Tokenizer

THIS_DIR = Path(__file__).parent


class Template:
    def __init__(
        self,
        role,
        template_name,
        data_provider=None,
        notes=None,
    ):
        self.role = role
        self.template_name = template_name
        self.data_provider = data_provider or ""
        self._notes = notes or ""

    @property
    def notes(self):
        default = "↵ represents newline"
        notes = default
        if self._notes:
            notes += "\n"
            notes += self._notes
        return notes


# Llama4 templates - similar to Llama3 but with python_list format
TEMPLATES = [
    Template(
        "user",
        "user-default",
        "user_default",
    ),
    Template(
        "user",
        "user-images",
        "user_images",
    ),
    Template("user", "user-interleaved-images", "user_interleaved_images"),
    Template(
        "assistant",
        "assistant-builtin-tool-call",
        "assistant_builtin_tool_call",
        "Notice <|python_tag|>",
    ),
    Template(
        "assistant",
        "assistant-custom-tool-call",
        "assistant_custom_tool_call",
        "Notice [func_name(param=value)] format",
    ),
    Template(
        "assistant",
        "assistant-default",
        "assistant_default",
    ),
    Template(
        "system",
        "system-builtin-and-custom-tools",
        "system_message_builtin_and_custom_tools",
    ),
    Template(
        "system",
        "system-builtin-tools-only",
        "system_message_builtin_tools_only",
    ),
    Template(
        "system",
        "system-custom-tools-only",
        "system_message_custom_tools_only",
    ),
    Template(
        "system",
        "system-default",
        "system_default",
    ),
    Template(
        "tool",
        "tool-success",
        "tool_success",
        "Note ipython header and [stdout]",
    ),
    Template(
        "tool",
        "tool-failure",
        "tool_failure",
        "Note ipython header and [stderr]",
    ),
]


class Llama4Interface:
    def __init__(self, tool_prompt_format: ToolPromptFormat = ToolPromptFormat.python_list):
        self.tokenizer = Tokenizer.get_instance()
        self.formatter = ChatFormat(self.tokenizer)
        self.tool_prompt_format = tool_prompt_format

    def get_tokens(self, messages: list[RawMessage]) -> list[int]:
        model_input = self.formatter.encode_dialog_prompt(
            messages,
            self.tool_prompt_format,
        )
        return model_input.tokens

    def tool_response_messages(self, *args, **kwargs):
        template = ToolResponseGenerator().gen(*args, **kwargs)
        return [
            RawMessage(
                role="tool",
                content=template.render(),
            )
        ]

    def system_messages(
        self,
        builtin_tools: list[BuiltinTool],
        custom_tools: list[ToolDefinition],
        instruction: str | None = None,
    ) -> list[RawMessage]:
        messages = []

        sys_content = ""

        # Handle builtin tools with builtin tool generator
        if builtin_tools:
            tool_gen = BuiltinToolGenerator()
            tool_template = tool_gen.gen(builtin_tools)
            sys_content += tool_template.render()
            sys_content += "\n"

        # Handle custom tools with Llama4's python list generator
        if custom_tools:
            if self.tool_prompt_format != ToolPromptFormat.python_list:
                raise ValueError(f"Llama4 only supports python_list tool prompt format, got {self.tool_prompt_format}")

            tool_gen = PythonListCustomToolGenerator()
            tool_template = tool_gen.gen(custom_tools, instruction)
            sys_content += tool_template.render()
        else:
            # If no custom tools but have instruction, add it
            if instruction:
                sys_content += instruction

        messages.append(RawMessage(role="system", content=sys_content.strip()))

        return messages

    def assistant_response_messages(
        self,
        content: str,
        stop_reason: StopReason,
        tool_call: ToolCall | None = None,
    ) -> list[RawMessage]:
        tool_calls = []
        if tool_call:
            tool_calls.append(tool_call)

        return [
            RawMessage(
                role="assistant",
                content=content,
                stop_reason=stop_reason,
                tool_calls=tool_calls,
            )
        ]

    def user_message(self, content: str) -> list[RawMessage]:
        return [RawMessage(role="user", content=content)]

    def display_message_as_tokens(self, message: RawMessage) -> None:
        tokens = self.formatter.encode_message(message, self.tool_prompt_format)[0]
        decoded = [self.tokenizer.decode([t]) for t in tokens]

        print(f"\n{colored(f'Message ({message.role}):', 'yellow')}")
        for i, (t, d) in enumerate(zip(tokens, decoded, strict=False)):
            color = "light_blue" if d.startswith("<|") and d.endswith("|>") else "white"
            print(f"{i:4d}: {t:6d} {colored(repr(d), color)}")


def list_jinja_templates() -> list[Template]:
    return TEMPLATES


def render_jinja_template(name: str, tool_prompt_format: ToolPromptFormat):
    # This would render templates - for now just return empty
    # Can be implemented later if needed for Llama4-specific templates
    return ""
