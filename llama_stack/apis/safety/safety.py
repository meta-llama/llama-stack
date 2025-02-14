# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from llama_stack.apis.inference import Message
from llama_stack.apis.shields import Shield
from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol
from llama_stack.schema_utils import json_schema_type, webmethod


@json_schema_type
class ViolationLevel(Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@json_schema_type
class SafetyViolation(BaseModel):
    violation_level: ViolationLevel

    # what message should you convey to the user
    user_message: Optional[str] = None

    # additional metadata (including specific violation codes) more for
    # debugging, telemetry
    metadata: Dict[str, Any] = Field(default_factory=dict)


@json_schema_type
class RunShieldResponse(BaseModel):
    violation: Optional[SafetyViolation] = None


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
        messages: List[Message],
        params: Dict[str, Any] = None,
    ) -> RunShieldResponse: ...
