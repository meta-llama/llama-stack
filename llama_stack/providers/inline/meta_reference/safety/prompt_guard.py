# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import auto, Enum
from typing import List

import torch

from llama_models.llama3.api.datatypes import Message
from termcolor import cprint

from .base import message_content_as_str, OnViolationAction, ShieldResponse, TextShield


class PromptGuardShield(TextShield):
    class Mode(Enum):
        INJECTION = auto()
        JAILBREAK = auto()

    _instances = {}
    _model_cache = None

    @staticmethod
    def instance(
        model_dir: str,
        threshold: float = 0.9,
        temperature: float = 1.0,
        mode: "PromptGuardShield.Mode" = Mode.JAILBREAK,
        on_violation_action=OnViolationAction.RAISE,
    ) -> "PromptGuardShield":
        action_value = on_violation_action.value
        key = (model_dir, threshold, temperature, mode, action_value)
        if key not in PromptGuardShield._instances:
            PromptGuardShield._instances[key] = PromptGuardShield(
                model_dir=model_dir,
                threshold=threshold,
                temperature=temperature,
                mode=mode,
                on_violation_action=on_violation_action,
            )
        return PromptGuardShield._instances[key]

    def __init__(
        self,
        model_dir: str,
        threshold: float = 0.9,
        temperature: float = 1.0,
        mode: "PromptGuardShield.Mode" = Mode.JAILBREAK,
        on_violation_action: OnViolationAction = OnViolationAction.RAISE,
    ):
        super().__init__(on_violation_action)
        assert (
            model_dir is not None
        ), "Must provide a model directory for prompt injection shield"
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0")
        self.device = "cuda"
        if PromptGuardShield._model_cache is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            # load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_dir, device_map=self.device
            )
            PromptGuardShield._model_cache = (tokenizer, model)

        self.tokenizer, self.model = PromptGuardShield._model_cache
        self.temperature = temperature
        self.threshold = threshold
        self.mode = mode

    def convert_messages_to_text(self, messages: List[Message]) -> str:
        return message_content_as_str(messages[-1])

    async def run_impl(self, text: str) -> ShieldResponse:
        # run model on messages and return response
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {name: tensor.to(self.model.device) for name, tensor in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs[0]
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        score_embedded = probabilities[0, 1].item()
        score_malicious = probabilities[0, 2].item()
        cprint(
            f"Ran PromptGuardShield and got Scores: Embedded: {score_embedded}, Malicious: {score_malicious}",
            color="magenta",
        )

        if self.mode == self.Mode.INJECTION and (
            score_embedded + score_malicious > self.threshold
        ):
            return ShieldResponse(
                is_violation=True,
                violation_type=f"prompt_injection:embedded={score_embedded},malicious={score_malicious}",
                violation_return_message="Sorry, I cannot do this.",
            )
        elif self.mode == self.Mode.JAILBREAK and score_malicious > self.threshold:
            return ShieldResponse(
                is_violation=True,
                violation_type=f"prompt_injection:malicious={score_malicious}",
                violation_return_message="Sorry, I cannot do this.",
            )

        return ShieldResponse(
            is_violation=False,
        )


class JailbreakShield(PromptGuardShield):
    def __init__(
        self,
        model_dir: str,
        threshold: float = 0.9,
        temperature: float = 1.0,
        on_violation_action: OnViolationAction = OnViolationAction.RAISE,
    ):
        super().__init__(
            model_dir=model_dir,
            threshold=threshold,
            temperature=temperature,
            mode=PromptGuardShield.Mode.JAILBREAK,
            on_violation_action=on_violation_action,
        )


class InjectionShield(PromptGuardShield):
    def __init__(
        self,
        model_dir: str,
        threshold: float = 0.9,
        temperature: float = 1.0,
        on_violation_action: OnViolationAction = OnViolationAction.RAISE,
    ):
        super().__init__(
            model_dir=model_dir,
            threshold=threshold,
            temperature=temperature,
            mode=PromptGuardShield.Mode.INJECTION,
            on_violation_action=on_violation_action,
        )
