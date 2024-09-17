# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_models.llama3.api.datatypes import Message, Role, UserMessage

from llama_stack.apis.safety import (
    OnViolationAction,
    RunShieldRequest,
    Safety,
    ShieldDefinition,
    ShieldResponse,
)
from termcolor import cprint


class SafetyException(Exception):  # noqa: N818
    def __init__(self, response: ShieldResponse):
        self.response = response
        super().__init__(response.violation_return_message)


class ShieldRunnerMixin:
    def __init__(
        self,
        safety_api: Safety,
        input_shields: List[ShieldDefinition] = None,
        output_shields: List[ShieldDefinition] = None,
    ):
        self.safety_api = safety_api
        self.input_shields = input_shields
        self.output_shields = output_shields

    async def run_shields(
        self, messages: List[Message], shields: List[ShieldDefinition]
    ) -> List[ShieldResponse]:
        messages = messages.copy()
        # some shields like llama-guard require the first message to be a user message
        # since this might be a tool call, first role might not be user
        if len(messages) > 0 and messages[0].role != Role.user.value:
            messages[0] = UserMessage(content=messages[0].content)

        res = await self.safety_api.run_shields(
            RunShieldRequest(
                messages=messages,
                shields=shields,
            )
        )

        results = res.responses
        for shield, r in zip(shields, results):
            if r.is_violation:
                if shield.on_violation_action == OnViolationAction.RAISE:
                    raise SafetyException(r)
                elif shield.on_violation_action == OnViolationAction.WARN:
                    cprint(
                        f"[Warn]{shield.__class__.__name__} raised a warning",
                        color="red",
                    )

        return results
