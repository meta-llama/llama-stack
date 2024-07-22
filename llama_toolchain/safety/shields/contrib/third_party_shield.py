import sys
from typing import List

from llama_models.llama3_1.api.datatypes import Message

parent_dir = "../.."
sys.path.append(parent_dir)
from llama_toolchain.safety.shields.base import OnViolationAction, ShieldBase, ShieldResponse

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
