# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
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
    api_dependencies: List[Api] = Field(
        default_factory=list,
        description="Higher-level API surfaces may depend on other providers to provide their functionality",
    )


@json_schema_type
class AdapterSpec(BaseModel):
    adapter_id: str = Field(
        ...,
        description="Unique identifier for this adapter",
    )
    module: str = Field(
        ...,
        description="""
Fully-qualified name of the module to import. The module is expected to have:

 - `get_adapter_impl(config, deps)`: returns the adapter implementation
""",
    )
    pip_packages: List[str] = Field(
        default_factory=list,
        description="The pip dependencies needed for this implementation",
    )
    config_class: Optional[str] = Field(
        default=None,
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


class RemoteProviderConfig(BaseModel):
    url: str = Field(..., description="The URL for the provider")

    @validator("url")
    @classmethod
    def validate_url(cls, url: str) -> str:
        if not url.startswith("http"):
            raise ValueError(f"URL must start with http: {url}")
        return url.rstrip("/")


def remote_provider_id(adapter_id: str) -> str:
    return f"remote::{adapter_id}"


@json_schema_type
class RemoteProviderSpec(ProviderSpec):
    adapter: Optional[AdapterSpec] = Field(
        default=None,
        description="""
If some code is needed to convert the remote responses into Llama Stack compatible
API responses, specify the adapter here. If not specified, it indicates the remote
as being "Llama Stack compatible"
""",
    )

    @property
    def docker_image(self) -> Optional[str]:
        return None

    @property
    def module(self) -> str:
        if self.adapter:
            return self.adapter.module
        return f"llama_toolchain.{self.api.value}.client"

    @property
    def pip_packages(self) -> List[str]:
        if self.adapter:
            return self.adapter.pip_packages
        return []


# Can avoid this by using Pydantic computed_field
def remote_provider_spec(
    api: Api, adapter: Optional[AdapterSpec] = None
) -> RemoteProviderSpec:
    config_class = (
        adapter.config_class
        if adapter and adapter.config_class
        else "llama_toolchain.core.datatypes.RemoteProviderConfig"
    )
    provider_id = remote_provider_id(adapter.adapter_id) if adapter else "remote"

    return RemoteProviderSpec(
        api=api, provider_id=provider_id, config_class=config_class, adapter=adapter
    )


@json_schema_type
class DistributionSpec(BaseModel):
    distribution_id: str
    description: str

    docker_image: Optional[str] = None
    providers: Dict[Api, str] = Field(
        default_factory=dict,
        description="Provider IDs for each of the APIs provided by this distribution",
    )


@json_schema_type
class PackageConfig(BaseModel):
    built_at: datetime

    package_name: str = Field(
        ...,
        description="""
Reference to the distribution this package refers to. For unregistered (adhoc) packages,
this could be just a hash
""",
    )
    distribution_id: Optional[str] = None

    docker_image: Optional[str] = Field(
        default=None,
        description="Reference to the docker image if this package refers to a container",
    )
    conda_env: Optional[str] = Field(
        default=None,
        description="Reference to the conda environment if this package refers to a conda environment",
    )
    providers: Dict[str, Any] = Field(
        default_factory=dict,
        description="""
Provider configurations for each of the APIs provided by this package. This includes configurations for
the dependencies of these providers as well.
""",
    )
