# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import logging

from typing import List

from llama_models.llama3.api.datatypes import Message

from llama_stack.apis.safety import *  # noqa: F403

log = logging.getLogger(__name__)


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
        self, messages: List[Message], identifiers: List[str]
    ) -> None:
        responses = await asyncio.gather(
            *[
                self.safety_api.run_shield(
                    shield_id=identifier,
                    messages=messages,
                )
                for identifier in identifiers
            ]
        )
        for identifier, response in zip(identifiers, responses):
            if not response.violation:
                continue

            violation = response.violation
            if violation.violation_level == ViolationLevel.ERROR:
                raise SafetyException(violation)
            elif violation.violation_level == ViolationLevel.WARN:
                log.warning(f"[Warn]{identifier} raised a warning")
