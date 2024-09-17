# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import re

from string import Template
from typing import List, Optional

import torch
from llama_models.llama3.api.datatypes import Message, Role
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import CANNED_RESPONSE_TEXT, OnViolationAction, ShieldBase, ShieldResponse
from llama_stack.apis.safety import *  # noqa: F403

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
    CAT_CODE_INTERPRETER_ABUSE,
]

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
    @staticmethod
    def instance(
        on_violation_action=OnViolationAction.RAISE,
        model_dir: str = None,
        excluded_categories: List[str] = None,
        disable_input_check: bool = False,
        disable_output_check: bool = False,
    ) -> "LlamaGuardShield":
        global _INSTANCE
        if _INSTANCE is None:
            _INSTANCE = LlamaGuardShield(
                on_violation_action,
                model_dir,
                excluded_categories,
                disable_input_check,
                disable_output_check,
            )
        return _INSTANCE

    def __init__(
        self,
        on_violation_action: OnViolationAction = OnViolationAction.RAISE,
        model_dir: str = None,
        excluded_categories: List[str] = None,
        disable_input_check: bool = False,
        disable_output_check: bool = False,
    ):
        super().__init__(on_violation_action)

        dtype = torch.bfloat16

        assert model_dir is not None, "Llama Guard model_dir is None"

        if excluded_categories is None:
            excluded_categories = []

        assert len(excluded_categories) == 0 or all(
            x in SAFETY_CATEGORIES_TO_CODE_MAP.values() for x in excluded_categories
        ), "Invalid categories in excluded categories. Expected format is ['S1', 'S2', ..]"

        self.device = "cuda"
        self.excluded_categories = excluded_categories
        self.disable_input_check = disable_input_check
        self.disable_output_check = disable_output_check

        # load model
        torch_dtype = torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch_dtype, device_map=self.device
        )

    def get_shield_type(self) -> ShieldType:
        return BuiltinShield.llama_guard

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

        categories = []
        for cat in DEFAULT_LG_V3_SAFETY_CATEGORIES:
            cat_code = SAFETY_CATEGORIES_TO_CODE_MAP[cat]
            if cat_code in excluded_categories:
                continue
            categories.append(f"{cat_code}: {cat}.")

        return categories

    def build_prompt(self, messages: List[Message]) -> str:
        categories = self.get_safety_categories()
        categories_str = "\n".join(categories)
        conversations_str = "\n\n".join(
            [f"{m.role.capitalize()}: {m.content}" for m in messages]
        )
        return PROMPT_TEMPLATE.substitute(
            agent_type=messages[-1].role.capitalize(),
            categories=categories_str,
            conversations=conversations_str,
        )

    def get_shield_response(self, response: str) -> ShieldResponse:
        if response == SAFE_RESPONSE:
            return ShieldResponse(
                shield_type=BuiltinShield.llama_guard, is_violation=False
            )
        unsafe_code = self.check_unsafe_response(response)
        if unsafe_code:
            unsafe_code_list = unsafe_code.split(",")
            if set(unsafe_code_list).issubset(set(self.excluded_categories)):
                return ShieldResponse(
                    shield_type=BuiltinShield.llama_guard, is_violation=False
                )
            return ShieldResponse(
                shield_type=BuiltinShield.llama_guard,
                is_violation=True,
                violation_type=unsafe_code,
                violation_return_message=CANNED_RESPONSE_TEXT,
            )

        raise ValueError(f"Unexpected response: {response}")

    async def run(self, messages: List[Message]) -> ShieldResponse:
        if self.disable_input_check and messages[-1].role == Role.user.value:
            return ShieldResponse(
                shield_type=BuiltinShield.llama_guard, is_violation=False
            )
        elif self.disable_output_check and messages[-1].role == Role.assistant.value:
            return ShieldResponse(
                shield_type=BuiltinShield.llama_guard,
                is_violation=False,
            )
        else:
            prompt = self.build_prompt(messages)
            llama_guard_input = {
                "role": "user",
                "content": prompt,
            }
            input_ids = self.tokenizer.apply_chat_template(
                [llama_guard_input], return_tensors="pt", tokenize=True
            ).to(self.device)
            prompt_len = input_ids.shape[1]
            output = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=20,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=0,
            )
            generated_tokens = output.sequences[:, prompt_len:]

            response = self.tokenizer.decode(
                generated_tokens[0], skip_special_tokens=True
            )
            response = response.strip()
            shield_response = self.get_shield_response(response)
            return shield_response
