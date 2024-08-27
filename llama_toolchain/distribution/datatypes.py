# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Dict, List, Optional

from llama_models.schema_utils import json_schema_type

from pydantic import BaseModel, Field, validator


@json_schema_type
class Api(Enum):
    inference = "inference"
    safety = "safety"
    agentic_system = "agentic_system"
    memory = "memory"


@json_schema_type
class ApiEndpoint(BaseModel):
    route: str
    method: str
    name: str


@json_schema_type
class ProviderSpec(BaseModel):
    api: Api
    provider_id: str
    config_class: str = Field(
        ...,
        description="Fully-qualified classname of the config for this provider",
    )


@json_schema_type
class InlineProviderSpec(ProviderSpec):
    pip_packages: List[str] = Field(
        default_factory=list,
        description="The pip dependencies needed for this implementation",
    )
    docker_image: Optional[str] = Field(
        default=None,
        description="""
The docker image to use for this implementation. If one is provided, pip_packages will be ignored.
If a provider depends on other providers, the dependencies MUST NOT specify a docker image.
""",
    )
    module: str = Field(
        ...,
        description="""
Fully-qualified name of the module to import. The module is expected to have:

 - `get_provider_impl(config, deps)`: returns the local implementation
""",
    )
    api_dependencies: List[Api] = Field(
        default_factory=list,
        description="Higher-level API surfaces may depend on other providers to provide their functionality",
    )


class RemoteProviderConfig(BaseModel):
    base_url: str = Field(..., description="The base URL for the llama stack provider")

    @validator("base_url")
    @classmethod
    def validate_base_url(cls, base_url: str) -> str:
        if not base_url.startswith("http"):
            raise ValueError(f"URL must start with http: {base_url}")
        return base_url


@json_schema_type
class RemoteProviderSpec(ProviderSpec):
    module: str = Field(
        ...,
        description="""
Fully-qualified name of the module to import. The module is expected to have:
 - `get_client_impl(base_url)`: returns a client which can be used to call the remote implementation
""",
    )
    config_class: str = "llama_toolchain.distribution.datatypes.RemoteProviderConfig"


@json_schema_type
class DistributionSpec(BaseModel):
    spec_id: str
    description: str

    provider_specs: Dict[Api, ProviderSpec] = Field(
        default_factory=dict,
        description="Provider specifications for each of the APIs provided by this distribution",
    )


@json_schema_type
class DistributionConfig(BaseModel):
    """References to a installed / configured DistributionSpec"""

    name: str
    spec: str
    conda_env: str
    providers: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provider configurations for each of the APIs provided by this distribution",
    )
