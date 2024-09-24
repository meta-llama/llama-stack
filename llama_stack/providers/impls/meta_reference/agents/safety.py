# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

from typing import List

from llama_models.llama3.api.datatypes import Message
from termcolor import cprint

from llama_stack.apis.safety import *  # noqa: F403


class SafetyException(Exception):  # noqa: N818
    def __init__(self, violation: SafetyViolation):
        self.violation = violation
        super().__init__(violation.user_message)


class ShieldRunnerMixin:
    def __init__(
        self,
        safety_api: Safety,
        input_shields: List[str] = None,
        output_shields: List[str] = None,
    ):
        self.safety_api = safety_api
        self.input_shields = input_shields
        self.output_shields = output_shields

    async def run_multiple_shields(
        self, messages: List[Message], shield_types: List[str]
    ) -> None:
        responses = await asyncio.gather(
            *[
                self.safety_api.run_shield(
                    shield_type=shield_type,
                    messages=messages,
                )
                for shield_type in shield_types
            ]
        )
        for shield_type, response in zip(shield_types, responses):
            if not response.violation:
                continue

            violation = response.violation
            if violation.violation_level == ViolationLevel.ERROR:
                raise SafetyException(violation)
            elif violation.violation_level == ViolationLevel.WARN:
                cprint(
                    f"[Warn]{shield_type} raised a warning",
                    color="red",
                )
