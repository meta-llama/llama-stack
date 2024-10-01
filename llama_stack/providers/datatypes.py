# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Union

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field


@json_schema_type
class Api(Enum):
    inference = "inference"
    safety = "safety"
    agents = "agents"
    memory = "memory"

    telemetry = "telemetry"

    models = "models"
    shields = "shields"
    memory_banks = "memory_banks"


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


class RoutingTable(Protocol):
    def get_routing_keys(self) -> List[str]: ...

    def get_provider_impl(self, routing_key: str) -> Any: ...


class RoutableProvider(Protocol):
    """
    A provider which sits behind the RoutingTable and can get routed to.

    All Inference / Safety / Memory providers fall into this bucket.
    """

    async def validate_routing_keys(self, keys: List[str]) -> None: ...


class GenericProviderConfig(BaseModel):
    provider_id: str
    config: Dict[str, Any]


class PlaceholderProviderConfig(BaseModel):
    """Placeholder provider config for API whose provider are defined in routing_table"""

    providers: List[str]


RoutingKey = Union[str, List[str]]


class RoutableProviderConfig(GenericProviderConfig):
    routing_key: RoutingKey


# Example: /inference, /safety
@json_schema_type
class AutoRoutedProviderSpec(ProviderSpec):
    provider_id: str = "router"
    config_class: str = ""

    docker_image: Optional[str] = None
    routing_table_api: Api
    module: str = Field(
        ...,
        description="""
        Fully-qualified name of the module to import. The module is expected to have:

        - `get_router_impl(config, provider_specs, deps)`: returns the router implementation
        """,
    )
    provider_data_validator: Optional[str] = Field(
        default=None,
    )

    @property
    def pip_packages(self) -> List[str]:
        raise AssertionError("Should not be called on AutoRoutedProviderSpec")


# Example: /models, /shields
@json_schema_type
class RoutingTableProviderSpec(ProviderSpec):
    provider_id: str = "routing_table"
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
    pip_packages: List[str] = Field(default_factory=list)


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
    provider_data_validator: Optional[str] = Field(
        default=None,
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
    provider_data_validator: Optional[str] = Field(
        default=None,
    )


class RemoteProviderConfig(BaseModel):
    host: str = "localhost"
    port: int

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


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

    @property
    def provider_data_validator(self) -> Optional[str]:
        if self.adapter:
            return self.adapter.provider_data_validator
        return None


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
