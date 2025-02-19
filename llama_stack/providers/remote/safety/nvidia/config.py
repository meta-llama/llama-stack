# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from enum import Enum
import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, SecretStr, field_validator

from llama_models.schema_utils import json_schema_type


class ShieldType(Enum):
    self_check = "self_check"


@json_schema_type
class NVIDIASafetyConfig(BaseModel):
    """
     Configuration for the NVIDIA Guardrail microservice endpoint.

    Attributes:
        url (str): A base url for accessing the NVIDIA guardrail endpoint, e.g. http://localhost:8000
        api_key (str): The access key for the hosted NIM endpoints

    There are two ways to access NVIDIA NIMs -
     0. Hosted: Preview APIs hosted at https://integrate.api.nvidia.com
     1. Self-hosted: You can run NVIDIA NIMs on your own infrastructure

    By default the configuration is set to use the hosted APIs. This requires
    an API key which can be obtained from https://ngc.nvidia.com/.

    By default the configuration will attempt to read the NVIDIA_API_KEY environment
    variable to set the api_key. Please do not put your API key in code.
    """
    guardrails_service_url: str = Field(
        default_factory=lambda: os.getenv("NVIDIA_BASE_URL", "https://0.0.0.0:7331"),
        description="The url for accessing the guardrails service",
    )
    config_id: Optional[str] = Field(
        default="self-check",
        description="Config ID to use from the config store"
    )
    config_store_path: Optional[str] = Field(
        default="/config-store",
        description="Path to config store"
    )

    @classmethod
    @field_validator("guard_type")
    def validate_guard_type(cls, v):
        if v not in [t.value for t in ShieldType]:
            raise ValueError(f"Unknown shield type: {v}")
        return v
    
    @classmethod
    def sample_run_config(cls, **kwargs) -> Dict[str, Any]:
        return {
            "guardrails_service_url": "${env.GUARDRAILS_SERVICE_URL:http://localhost:7331}",
            "config_id": "self-check"
        }
