# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Dict, Optional, Union

from llama_models.llama3_1.api.datatypes import ToolParamDefinition

from llama_models.schema_utils import json_schema_type

from pydantic import BaseModel

from llama_toolchain.common.deployment_types import RestAPIExecutionConfig


@json_schema_type
class BuiltinShield(Enum):
    llama_guard = "llama_guard"
    code_scanner_guard = "code_scanner_guard"
    third_party_shield = "third_party_shield"
    injection_shield = "injection_shield"
    jailbreak_shield = "jailbreak_shield"


ShieldType = Union[BuiltinShield, str]


@json_schema_type
class OnViolationAction(Enum):
    IGNORE = 0
    WARN = 1
    RAISE = 2


@json_schema_type
class ShieldDefinition(BaseModel):
    shield_type: ShieldType
    description: Optional[str] = None
    parameters: Optional[Dict[str, ToolParamDefinition]] = None
    on_violation_action: OnViolationAction = OnViolationAction.RAISE
    execution_config: Optional[RestAPIExecutionConfig] = None


@json_schema_type
class ShieldResponse(BaseModel):
    shield_type: ShieldType
    # TODO(ashwin): clean this up
    is_violation: bool
    violation_type: Optional[str] = None
    violation_return_message: Optional[str] = None
