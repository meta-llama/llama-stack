# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import re

from string import Template
from typing import List, Optional

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.inference import *  # noqa: F403

from .base import CANNED_RESPONSE_TEXT, OnViolationAction, ShieldBase, ShieldResponse


SAFE_RESPONSE = "safe"
_INSTANCE = None

CAT_VIOLENT_CRIMES = "Violent Crimes"
CAT_NON_VIOLENT_CRIMES = "Non-Violent Crimes"
CAT_SEX_CRIMES = "Sex Crimes"
CAT_CHILD_EXPLOITATION = "Child Exploitation"
CAT_DEFAMATION = "Defamation"
CAT_SPECIALIZED_ADVICE = "Specialized Advice"
CAT_PRIVACY = "Privacy"
CAT_INTELLECTUAL_PROPERTY = "Intellectual Property"
CAT_INDISCRIMINATE_WEAPONS = "Indiscriminate Weapons"
CAT_HATE = "Hate"
CAT_SELF_HARM = "Self-Harm"
CAT_SEXUAL_CONTENT = "Sexual Content"
CAT_ELECTIONS = "Elections"
CAT_CODE_INTERPRETER_ABUSE = "Code Interpreter Abuse"


SAFETY_CATEGORIES_TO_CODE_MAP = {
    CAT_VIOLENT_CRIMES: "S1",
    CAT_NON_VIOLENT_CRIMES: "S2",
    CAT_SEX_CRIMES: "S3",
    CAT_CHILD_EXPLOITATION: "S4",
    CAT_DEFAMATION: "S5",
    CAT_SPECIALIZED_ADVICE: "S6",
    CAT_PRIVACY: "S7",
    CAT_INTELLECTUAL_PROPERTY: "S8",
    CAT_INDISCRIMINATE_WEAPONS: "S9",
    CAT_HATE: "S10",
    CAT_SELF_HARM: "S11",
    CAT_SEXUAL_CONTENT: "S12",
    CAT_ELECTIONS: "S13",
    CAT_CODE_INTERPRETER_ABUSE: "S14",
}


DEFAULT_LG_V3_SAFETY_CATEGORIES = [
    CAT_VIOLENT_CRIMES,
    CAT_NON_VIOLENT_CRIMES,
    CAT_SEX_CRIMES,
    CAT_CHILD_EXPLOITATION,
    CAT_DEFAMATION,
    CAT_SPECIALIZED_ADVICE,
    CAT_PRIVACY,
    CAT_INTELLECTUAL_PROPERTY,
    CAT_INDISCRIMINATE_WEAPONS,
    CAT_HATE,
    CAT_SELF_HARM,
    CAT_SEXUAL_CONTENT,
    CAT_ELECTIONS,
]


MODEL_TO_SAFETY_CATEGORIES_MAP = {
    CoreModelId.llama_guard_3_8b.value: (
        DEFAULT_LG_V3_SAFETY_CATEGORIES + [CAT_CODE_INTERPRETER_ABUSE]
    ),
    CoreModelId.llama_guard_3_1b.value: DEFAULT_LG_V3_SAFETY_CATEGORIES,
    CoreModelId.llama_guard_3_11b_vision.value: DEFAULT_LG_V3_SAFETY_CATEGORIES,
}


PROMPT_TASK = "Task: Check if there is unsafe content in '$agent_type' messages in conversations according our safety policy with the below categories."

SAFETY_CATEGORIES = """

<BEGIN UNSAFE CONTENT CATEGORIES>
$categories
<END UNSAFE CONTENT CATEGORIES>
"""

PROMPT_CONVERSATION = """
<BEGIN CONVERSATION>

$conversations

<END CONVERSATION>
"""


PROMPT_INSTRUCTIONS = """
Provide your safety assessment for ONLY THE LAST $agent_type message in the above conversation:
 - First line must read 'safe' or 'unsafe'.
 - If unsafe, a second line must include a comma-separated list of violated categories."""


PROMPT_TEMPLATE = Template(
    f"{PROMPT_TASK}{SAFETY_CATEGORIES}{PROMPT_CONVERSATION}{PROMPT_INSTRUCTIONS}"
)


class LlamaGuardShield(ShieldBase):
    def __init__(
        self,
        model: str,
        inference_api: Inference,
        excluded_categories: List[str] = None,
        disable_input_check: bool = False,
        disable_output_check: bool = False,
        on_violation_action: OnViolationAction = OnViolationAction.RAISE,
    ):
        super().__init__(on_violation_action)

        if excluded_categories is None:
            excluded_categories = []

        assert len(excluded_categories) == 0 or all(
            x in SAFETY_CATEGORIES_TO_CODE_MAP.values() for x in excluded_categories
        ), "Invalid categories in excluded categories. Expected format is ['S1', 'S2', ..]"

        if model not in MODEL_TO_SAFETY_CATEGORIES_MAP:
            raise ValueError(f"Unsupported model: {model}")

        self.model = model
        self.inference_api = inference_api
        self.excluded_categories = excluded_categories
        self.disable_input_check = disable_input_check
        self.disable_output_check = disable_output_check

    def check_unsafe_response(self, response: str) -> Optional[str]:
        match = re.match(r"^unsafe\n(.*)$", response)
        if match:
            # extracts the unsafe code
            extracted = match.group(1)
            return extracted

        return None

    def get_safety_categories(self) -> List[str]:
        excluded_categories = self.excluded_categories
        if set(excluded_categories) == set(SAFETY_CATEGORIES_TO_CODE_MAP.values()):
            excluded_categories = []

        final_categories = []

        all_categories = MODEL_TO_SAFETY_CATEGORIES_MAP[self.model]
        for cat in all_categories:
            cat_code = SAFETY_CATEGORIES_TO_CODE_MAP[cat]
            if cat_code in excluded_categories:
                continue
            final_categories.append(f"{cat_code}: {cat}.")

        return final_categories

    def validate_messages(self, messages: List[Message]) -> None:
        if len(messages) == 0:
            raise ValueError("Messages must not be empty")
        if messages[0].role != Role.user.value:
            raise ValueError("Messages must start with user")

        if len(messages) >= 2 and (
            messages[0].role == Role.user.value and messages[1].role == Role.user.value
        ):
            messages = messages[1:]

        for i in range(1, len(messages)):
            if messages[i].role == messages[i - 1].role:
                raise ValueError(
                    f"Messages must alternate between user and assistant. Message {i} has the same role as message {i-1}"
                )
        return messages

    async def run(self, messages: List[Message]) -> ShieldResponse:
        messages = self.validate_messages(messages)
        if self.disable_input_check and messages[-1].role == Role.user.value:
            return ShieldResponse(is_violation=False)
        elif self.disable_output_check and messages[-1].role == Role.assistant.value:
            return ShieldResponse(
                is_violation=False,
            )

        if self.model == CoreModelId.llama_guard_3_11b_vision.value:
            shield_input_message = self.build_vision_shield_input(messages)
        else:
            shield_input_message = self.build_text_shield_input(messages)

        # TODO: llama-stack inference protocol has issues with non-streaming inference code
        content = ""
        async for chunk in self.inference_api.chat_completion(
            model=self.model,
            messages=[shield_input_message],
            stream=True,
        ):
            event = chunk.event
            if event.event_type == ChatCompletionResponseEventType.progress:
                assert isinstance(event.delta, str)
                content += event.delta

        content = content.strip()
        shield_response = self.get_shield_response(content)
        return shield_response

    def build_text_shield_input(self, messages: List[Message]) -> UserMessage:
        return UserMessage(content=self.build_prompt(messages))

    def build_vision_shield_input(self, messages: List[Message]) -> UserMessage:
        conversation = []
        most_recent_img = None

        for m in messages[::-1]:
            if isinstance(m.content, str):
                conversation.append(m)
            elif isinstance(m.content, ImageMedia):
                if most_recent_img is None and m.role == Role.user.value:
                    most_recent_img = m.content
                    conversation.append(m)
            elif isinstance(m.content, list):
                content = []
                for c in m.content:
                    if isinstance(c, str):
                        content.append(c)
                    elif isinstance(c, ImageMedia):
                        if most_recent_img is None and m.role == Role.user.value:
                            most_recent_img = c
                            content.append(c)
                    else:
                        raise ValueError(f"Unknown content type: {c}")

                conversation.append(UserMessage(content=content))
            else:
                raise ValueError(f"Unknown content type: {m.content}")

        prompt = []
        if most_recent_img is not None:
            prompt.append(most_recent_img)
        prompt.append(self.build_prompt(conversation[::-1]))

        return UserMessage(content=prompt)

    def build_prompt(self, messages: List[Message]) -> str:
        categories = self.get_safety_categories()
        categories_str = "\n".join(categories)
        conversations_str = "\n\n".join(
            [
                f"{m.role.capitalize()}: {interleaved_text_media_as_str(m.content)}"
                for m in messages
            ]
        )
        return PROMPT_TEMPLATE.substitute(
            agent_type=messages[-1].role.capitalize(),
            categories=categories_str,
            conversations=conversations_str,
        )

    def get_shield_response(self, response: str) -> ShieldResponse:
        response = response.strip()
        if response == SAFE_RESPONSE:
            return ShieldResponse(is_violation=False)
        unsafe_code = self.check_unsafe_response(response)
        if unsafe_code:
            unsafe_code_list = unsafe_code.split(",")
            if set(unsafe_code_list).issubset(set(self.excluded_categories)):
                return ShieldResponse(is_violation=False)
            return ShieldResponse(
                is_violation=True,
                violation_type=unsafe_code,
                violation_return_message=CANNED_RESPONSE_TEXT,
            )

        raise ValueError(f"Unexpected response: {response}")
