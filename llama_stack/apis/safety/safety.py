# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from llama_stack.apis.inference import Message
from llama_stack.apis.shields import Shield
from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol
from llama_stack.schema_utils import json_schema_type, webmethod


@json_schema_type
class ViolationLevel(Enum):
    """Severity level of a safety violation.

    :cvar INFO: Informational level violation that does not require action
    :cvar WARN: Warning level violation that suggests caution but allows continuation
    :cvar ERROR: Error level violation that requires blocking or intervention
    """

    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@json_schema_type
class SafetyViolation(BaseModel):
    """Details of a safety violation detected by content moderation.

    :param violation_level: Severity level of the violation
    :param user_message: (Optional) Message to convey to the user about the violation
    :param metadata: Additional metadata including specific violation codes for debugging and telemetry
    """

    violation_level: ViolationLevel

    # what message should you convey to the user
    user_message: str | None = None

    # additional metadata (including specific violation codes) more for
    # debugging, telemetry
    metadata: dict[str, Any] = Field(default_factory=dict)


@json_schema_type
class RunShieldResponse(BaseModel):
    """Response from running a safety shield.

    :param violation: (Optional) Safety violation detected by the shield, if any
    """

    violation: SafetyViolation | None = None


class ShieldStore(Protocol):
    async def get_shield(self, identifier: str) -> Shield: ...


@runtime_checkable
@trace_protocol
class Safety(Protocol):
    shield_store: ShieldStore

    @webmethod(route="/safety/run-shield", method="POST")
    async def run_shield(
        self,
        shield_id: str,
        messages: list[Message],
        params: dict[str, Any],
    ) -> RunShieldResponse:
        """Run a shield.

        :param shield_id: The identifier of the shield to run.
        :param messages: The messages to run the shield on.
        :param params: The parameters of the shield.
        :returns: A RunShieldResponse.
        """
        ...
