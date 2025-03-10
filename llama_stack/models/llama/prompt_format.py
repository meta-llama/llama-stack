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

import json
import textwrap
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from llama_stack.models.llama.datatypes import (
    RawContent,
    RawMediaItem,
    RawMessage,
    RawTextItem,
    StopReason,
    ToolCall,
    ToolPromptFormat,
)

from .llama3.interface import LLama31Interface
from .llama3.template_data import (
    system_message_builtin_code_only,
    system_message_builtin_tools_only,
    system_message_custom_tools_only,
)


class TextCompletionContent(BaseModel):
    content: RawContent = ""


class UseCase(BaseModel):
    title: str = ""
    description: str = ""
    dialogs: List[List[RawMessage] | TextCompletionContent | str] = Field(default_factory=list)
    notes: str = ""
    tool_prompt_format: ToolPromptFormat = ToolPromptFormat.json

    def md_format(self):
        section = textwrap.dedent(
            """
            ## {title}

            {description}

            {dialogs_text}
            {notes}

            """
        )
        return section.lstrip()

    def dialogs_to_text(self, generator) -> str:
        def _code_block(text):
            return f"```\n{text}\n```"

        text = ""
        for dialog in self.dialogs:
            if isinstance(dialog, str):
                text += dialog
                text += "\n\n"
                continue

            elif isinstance(dialog, TextCompletionContent):
                input_tokens, output_tokens = generator.text_completion_raw(
                    dialog.content,
                    max_gen_len=64,
                    temperature=0.1,
                    top_p=0.95,
                )
            else:
                input_tokens, output_tokens = generator.chat_completion_raw(
                    dialog,
                    max_gen_len=512,
                    temperature=0.0,
                    top_p=0.95,
                    tool_prompt_format=self.tool_prompt_format,
                )
            text += "##### Input Prompt Format\n"

            # FIXME: This is added to undo the hack in chat_formatter where
            # vision tokens are replaced with 128256.
            input_tokens = [generator.formatter.vision_token if t == 128256 else t for t in input_tokens]

            text += _code_block(generator.tokenizer.decode(input_tokens))
            # TODO: Figure out if "↵" needs to be added for newlines or end or some indication
            text += "\n\n"
            text += "##### Model Response Format\n"
            text += _code_block(generator.tokenizer.decode(output_tokens))
            text += "\n\n"

        return text

    def to_text(self, generator):
        section = self.md_format()
        dialogs_text = self.dialogs_to_text(generator)
        notes = f"##### Notes\n{self.notes}" if self.notes else ""
        section = section.format(
            title=self.title,
            description=self.description,
            dialogs_text=dialogs_text,
            notes=notes,
        )
        return section


def llama3_1_builtin_tool_call_dialog(tool_prompt_format=ToolPromptFormat.json):
    interface = LLama31Interface(tool_prompt_format)

    messages = interface.system_messages(**system_message_builtin_tools_only())
    messages += interface.user_message(content="Search the web for the latest price of 1oz gold?")

    return messages


def llama3_1_builtin_code_interpreter_dialog(tool_prompt_format=ToolPromptFormat.json):
    interface = LLama31Interface(tool_prompt_format)

    messages = interface.system_messages(**system_message_builtin_code_only())
    messages += interface.user_message(
        content="Write code to check if number is prime. Use it to verify if number 7 is prime"
    )

    return messages


def llama3_1_builtin_tool_call_with_image_dialog(
    tool_prompt_format=ToolPromptFormat.json,
):
    this_dir = Path(__file__).parent
    with open(this_dir / "llama3/dog.jpg", "rb") as f:
        img = f.read()

    interface = LLama31Interface(tool_prompt_format)

    messages = interface.system_messages(**system_message_builtin_tools_only())
    messages += interface.user_message(content=[RawMediaItem(data=img), RawTextItem(text="What is this dog breed?")])
    messages += interface.assistant_response_messages(
        "Based on the description of the dog in the image, it appears to be a small breed dog, possibly a terrier mix",
        StopReason.end_of_turn,
    )
    messages += interface.user_message("Search the web for some food recommendations for the indentified breed")
    return messages


def llama3_1_custom_tool_call_dialog(tool_prompt_format=ToolPromptFormat.json):
    interface = LLama31Interface(tool_prompt_format)

    messages = interface.system_messages(**system_message_custom_tools_only())
    messages += interface.user_message(content="Use tools to get latest trending songs")
    return messages


def llama3_1_e2e_tool_call_dialog(tool_prompt_format=ToolPromptFormat.json):
    tool_response = json.dumps(["great song1", "awesome song2", "cool song3"])
    interface = LLama31Interface(tool_prompt_format)

    messages = interface.system_messages(**system_message_custom_tools_only())
    messages += interface.user_message(content="Use tools to get latest trending songs")
    messages.append(
        RawMessage(
            role="assistant",
            content="",
            stop_reason=StopReason.end_of_message,
            tool_calls=[
                ToolCall(
                    call_id="call_id",
                    tool_name="trending_songs",
                    arguments={"n": "10", "genre": "latest"},
                )
            ],
        ),
    )
    messages.append(
        RawMessage(
            role="assistant",
            content=tool_response,
        )
    )
    return messages


def llama3_2_user_assistant_conversation():
    return UseCase(
        title="User and assistant conversation",
        description="Here is a regular multi-turn user assistant conversation and how its formatted.",
        dialogs=[
            [
                RawMessage(role="system", content="You are a helpful assistant"),
                RawMessage(role="user", content="Who are you?"),
            ]
        ],
        notes="This format is unchanged from Llama3.1",
    )
