# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import List, Union

from llama_models.llama3_1.api.datatypes import Attachment, Message
from llama_toolchain.safety.api.datatypes import *  # noqa: F403

CANNED_RESPONSE_TEXT = "I can't answer that. Can I help with something else?"


class ShieldBase(ABC):

    def __init__(
        self,
        on_violation_action: OnViolationAction = OnViolationAction.RAISE,
    ):
        self.on_violation_action = on_violation_action

    @abstractmethod
    def get_shield_type(self) -> ShieldType:
        raise NotImplementedError()

    @abstractmethod
    async def run(self, messages: List[Message]) -> ShieldResponse:
        raise NotImplementedError()


def message_content_as_str(message: Message) -> str:
    def _to_str(content: Union[str, Attachment]) -> str:
        if isinstance(content, str):
            return content
        elif isinstance(content, Attachment):
            return f"File: {str(content.url)}"
        else:
            raise

    if isinstance(message.content, list) or isinstance(message.content, tuple):
        return "\n".join([_to_str(c) for c in message.content])
    else:
        return _to_str(message.content)


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

    def get_shield_type(self) -> ShieldType:
        return "dummy"

    async def run_impl(self, text: str) -> ShieldResponse:
        # Dummy return LOW to test e2e
        return ShieldResponse(
            shield_type=BuiltinShield.third_party_shield, is_violation=False
        )
