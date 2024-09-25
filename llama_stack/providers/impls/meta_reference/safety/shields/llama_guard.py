# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import re

from string import Template
from typing import List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    MllamaForConditionalGeneration,
    MllamaProcessor,
)

from .base import CANNED_RESPONSE_TEXT, OnViolationAction, ShieldBase, ShieldResponse
from llama_models.llama3.api.datatypes import Message, Role


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

# model names
LG_3_8B = "Llama-Guard-3-8B"
LG_3_1B = "Llama-Guard-3-1B"
LG_3_11B_VISION = "Llama-Guard-3-11B-Vision"


MODEL_TO_SAFETY_CATEGORIES_MAP = {
    LG_3_8B: DEFAULT_LG_V3_SAFETY_CATEGORIES + [CAT_CODE_INTERPRETER_ABUSE],
    LG_3_1B: DEFAULT_LG_V3_SAFETY_CATEGORIES,
    LG_3_11B_VISION: DEFAULT_LG_V3_SAFETY_CATEGORIES,
}


PROMPT_TASK = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nTask: Check if there is unsafe content in '$agent_type' messages in conversations according our safety policy with the below categories."

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
 - If unsafe, a second line must include a comma-separated list of violated categories. <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


PROMPT_TEMPLATE = Template(
    f"{PROMPT_TASK}{SAFETY_CATEGORIES}{PROMPT_CONVERSATION}{PROMPT_INSTRUCTIONS}"
)


class LlamaGuardShield(ShieldBase):
    def __init__(
        self,
        model_dir: str,
        excluded_categories: List[str] = None,
        disable_input_check: bool = False,
        disable_output_check: bool = False,
        on_violation_action: OnViolationAction = OnViolationAction.RAISE,
    ):
        super().__init__(on_violation_action)

        dtype = torch.bfloat16
        self.model_dir = model_dir
        self.device = "cuda"

        assert self.model_dir is not None, "Llama Guard model_dir is None"

        if excluded_categories is None:
            excluded_categories = []

        assert len(excluded_categories) == 0 or all(
            x in SAFETY_CATEGORIES_TO_CODE_MAP.values() for x in excluded_categories
        ), "Invalid categories in excluded categories. Expected format is ['S1', 'S2', ..]"

        self.excluded_categories = excluded_categories
        self.disable_input_check = disable_input_check
        self.disable_output_check = disable_output_check

        torch_dtype = torch.bfloat16

        if self.is_lg_vision():

            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_dir, device_map=self.device, torch_dtype=torch_dtype
            )
            self.processor = MllamaProcessor.from_pretrained(self.model_dir)
        else:

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir, torch_dtype=torch_dtype, device_map=self.device
            )

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
        all_categories = MODEL_TO_SAFETY_CATEGORIES_MAP[self.get_model_name()]
        for cat in all_categories:
            cat_code = SAFETY_CATEGORIES_TO_CODE_MAP[cat]
            if cat_code in excluded_categories:
                continue
            final_categories.append(f"{cat_code}: {cat}.")

        return final_categories

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

    def build_mm_prompt(self, messages: List[Message]) -> str:
        conversation = []
        most_recent_img = None

        for m in messages[::-1]:
            if isinstance(m.content, str):
                conversation.append(
                    {
                        "role": m.role,
                        "content": [{"type": "text", "text": m.content}],
                    }
                )
            elif isinstance(m.content, ImageMedia):
                if most_recent_img is None and m.role == Role.user.value:
                    most_recent_img = m.content
                    conversation.append(
                        {
                            "role": m.role,
                            "content": [{"type": "image"}],
                        }
                    )

            elif isinstance(m.content, list):
                content = []
                for c in m.content:
                    if isinstance(c, str):
                        content.append({"type": "text", "text": c})
                    elif isinstance(c, ImageMedia):
                        if most_recent_img is None and m.role == Role.user.value:
                            most_recent_img = c
                            content.append({"type": "image"})
                    else:
                        raise ValueError(f"Unknown content type: {c}")

                conversation.append(
                    {
                        "role": m.role,
                        "content": content,
                    }
                )
            else:
                raise ValueError(f"Unknown content type: {m.content}")

        return conversation[::-1], most_recent_img

    async def run_lg_mm(self, messages: List[Message]) -> ShieldResponse:
        formatted_messages, most_recent_img = self.build_mm_prompt(messages)
        raw_image = None
        if most_recent_img:
            raw_image = interleaved_text_media_localize(most_recent_img)
            raw_image = raw_image.image
        llama_guard_input_templ_applied = self.processor.apply_chat_template(
            formatted_messages,
            add_generation_prompt=True,
            tokenize=False,
            skip_special_tokens=False,
        )
        inputs = self.processor(
            text=llama_guard_input_templ_applied, images=raw_image, return_tensors="pt"
        ).to(self.device)
        output = self.model.generate(**inputs, do_sample=False, max_new_tokens=50)
        response = self.processor.decode(
            output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
        )
        shield_response = self.get_shield_response(response)
        return shield_response

    async def run_lg_text(self, messages: List[Message]):
        prompt = self.build_prompt(messages)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = input_ids.shape[1]
        output = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=20,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=0,
        )
        generated_tokens = output.sequences[:, prompt_len:]

        response = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        shield_response = self.get_shield_response(response)
        return shield_response

    def get_model_name(self):
        return self.model_dir.split("/")[-1]

    def is_lg_vision(self):
        model_name = self.get_model_name()
        return model_name == LG_3_11B_VISION

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
        else:

            if self.is_lg_vision():

                shield_response = await self.run_lg_mm(messages)

            else:

                shield_response = await self.run_lg_text(messages)

        return shield_response
