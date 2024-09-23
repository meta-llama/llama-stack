# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import List

from llama_models.llama3.api.datatypes import interleaved_text_media_as_str, Message
from pydantic import BaseModel
from llama_stack.apis.safety import *  # noqa: F403

CANNED_RESPONSE_TEXT = "I can't answer that. Can I help with something else?"


# TODO: clean this up; just remove this type completely
class ShieldResponse(BaseModel):
    is_violation: bool
    violation_type: Optional[str] = None
    violation_return_message: Optional[str] = None


# TODO: this is a caller / agent concern
class OnViolationAction(Enum):
    IGNORE = 0
    WARN = 1
    RAISE = 2


class ShieldBase(ABC):
    def __init__(
        self,
        on_violation_action: OnViolationAction = OnViolationAction.RAISE,
    ):
        self.on_violation_action = on_violation_action

    @abstractmethod
    async def run(self, messages: List[Message]) -> ShieldResponse:
        raise NotImplementedError()


def message_content_as_str(message: Message) -> str:
    return interleaved_text_media_as_str(message.content)


# For shields that operate on simple strings
class TextShield(ShieldBase):
    def convert_messages_to_text(self, messages: List[Message]) -> str:
        return "\n".join([message_content_as_str(m) for m in messages])

    async def run(self, messages: List[Message]) -> ShieldResponse:
        text = self.convert_messages_to_text(messages)
        return await self.run_impl(text)

    @abstractmethod
    async def run_impl(self, text: str) -> ShieldResponse:
        raise NotImplementedError()


class DummyShield(TextShield):
    async def run_impl(self, text: str) -> ShieldResponse:
        # Dummy return LOW to test e2e
        return ShieldResponse(is_violation=False)
