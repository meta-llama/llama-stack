# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from typing import List

from llama_models.llama3_1.api.datatypes import Message, Role

from .base import OnViolationAction, ShieldBase, ShieldResponse


class SafetyException(Exception):  # noqa: N818
    def __init__(self, response: ShieldResponse):
        self.response = response
        super().__init__(response.violation_return_message)


class ShieldRunnerMixin:

    def __init__(
        self,
        input_shields: List[ShieldBase] = None,
        output_shields: List[ShieldBase] = None,
    ):
        self.input_shields = input_shields
        self.output_shields = output_shields

    async def run_shields(
        self, messages: List[Message], shields: List[ShieldBase]
    ) -> List[ShieldResponse]:
        # some shields like llama-guard require the first message to be a user message
        # since this might be a tool call, first role might not be user
        if len(messages) > 0 and messages[0].role != Role.user.value:
            # TODO(ashwin): we need to change the type of the message, this kind of modification
            # is no longer appropriate
            messages[0].role = Role.user.value

        results = await asyncio.gather(*[s.run(messages) for s in shields])
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
