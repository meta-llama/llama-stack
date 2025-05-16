# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any

from pydantic import BaseModel, field_validator


class PromptGuardType(Enum):
    injection = "injection"
    jailbreak = "jailbreak"


class PromptGuardExecutionType(Enum):
    cpu = "cpu"
    cuda = "cuda"


class PromptGuardConfig(BaseModel):
    guard_type: str = PromptGuardType.injection.value
    guard_execution_type: str = PromptGuardExecutionType.cuda.value

    @classmethod
    @field_validator("guard_type")
    def validate_guard_type(cls, v):
        if v not in [t.value for t in PromptGuardType]:
            raise ValueError(f"Unknown prompt guard type: {v}")
        return v

    @classmethod
    @field_validator("guard_execution_type")
    def validate_guard_execution_type(cls, v):
        if v not in [t.value for t in PromptGuardExecutionType]:
            raise ValueError(f"Unknown prompt guard execution type: {v}")
        return v

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "guard_type": "injection",
            "guard_execution_type": "cuda",
        }
