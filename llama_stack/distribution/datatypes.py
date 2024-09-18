# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from llama_models.schema_utils import json_schema_type

from pydantic import BaseModel, Field, validator


@json_schema_type
class Api(Enum):
    inference = "inference"
    safety = "safety"
    agents = "agents"
    memory = "memory"
    telemetry = "telemetry"


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
class RouterProviderSpec(ProviderSpec):
    provider_id: str = "router"
    config_class: str = ""

    docker_image: Optional[str] = None

    inner_specs: List[ProviderSpec]
    module: str = Field(
        ...,
        description="""
Fully-qualified name of the module to import. The module is expected to have:

 - `get_router_impl(config, provider_specs, deps)`: returns the router implementation
""",
    )

    @property
    def pip_packages(self) -> List[str]:
        raise AssertionError("Should not be called on RouterProviderSpec")


class GenericProviderConfig(BaseModel):
    provider_id: str
    config: Dict[str, Any]


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
        return f"llama_stack.apis.{self.api.value}.client"

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
        else "llama_stack.distribution.datatypes.RemoteProviderConfig"
    )
    provider_id = remote_provider_id(adapter.adapter_id) if adapter else "remote"

    return RemoteProviderSpec(
        api=api, provider_id=provider_id, config_class=config_class, adapter=adapter
    )


@json_schema_type
class DistributionSpec(BaseModel):
    description: Optional[str] = Field(
        default="",
        description="Description of the distribution",
    )
    docker_image: Optional[str] = None
    providers: Dict[str, Union[str, List[str]]] = Field(
        default_factory=dict,
        description="""
Provider Types for each of the APIs provided by this distribution. If you
select multiple providers, you should provide an appropriate 'routing_map'
in the runtime configuration to help route to the correct provider.""",
    )


@json_schema_type
class ProviderRoutingEntry(GenericProviderConfig):
    routing_key: str


ProviderMapEntry = Union[GenericProviderConfig, List[ProviderRoutingEntry]]


@json_schema_type
class StackRunConfig(BaseModel):
    built_at: datetime

    image_name: str = Field(
        ...,
        description="""
Reference to the distribution this package refers to. For unregistered (adhoc) packages,
this could be just a hash
""",
    )
    docker_image: Optional[str] = Field(
        default=None,
        description="Reference to the docker image if this package refers to a container",
    )
    conda_env: Optional[str] = Field(
        default=None,
        description="Reference to the conda environment if this package refers to a conda environment",
    )
    apis_to_serve: List[str] = Field(
        description="""
The list of APIs to serve. If not specified, all APIs specified in the provider_map will be served""",
    )
    provider_map: Dict[str, ProviderMapEntry] = Field(
        description="""
Provider configurations for each of the APIs provided by this package.

Given an API, you can specify a single provider or a "routing table". Each entry in the routing
table has a (routing_key, provider_config) tuple. How the key is interpreted is API-specific.

As examples:
- the "inference" API interprets the routing_key as a "model"
- the "memory" API interprets the routing_key as the type of a "memory bank"

The key may support wild-cards alsothe routing_key to route to the correct provider.""",
    )


@json_schema_type
class BuildConfig(BaseModel):
    name: str
    distribution_spec: DistributionSpec = Field(
        description="The distribution spec to build including API providers. "
    )
    image_type: str = Field(
        default="conda",
        description="Type of package to build (conda | container)",
    )
