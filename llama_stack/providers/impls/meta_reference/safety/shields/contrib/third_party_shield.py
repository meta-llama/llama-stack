# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_models.llama3.api.datatypes import Message

from llama_stack.providers.impls.meta_reference.safety.shields.base import (
    OnViolationAction,
    ShieldBase,
    ShieldResponse,
)

_INSTANCE = None


class ThirdPartyShield(ShieldBase):
    @staticmethod
    def instance(on_violation_action=OnViolationAction.RAISE) -> "ThirdPartyShield":
        global _INSTANCE
        if _INSTANCE is None:
            _INSTANCE = ThirdPartyShield(on_violation_action)
        return _INSTANCE

    def __init__(
        self,
        on_violation_action: OnViolationAction = OnViolationAction.RAISE,
    ):
        super().__init__(on_violation_action)

    async def run(self, messages: List[Message]) -> ShieldResponse:
        super.run()  # will raise NotImplementedError
