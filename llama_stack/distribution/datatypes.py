# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator

from llama_stack.apis.benchmarks import Benchmark, BenchmarkInput
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Dataset, DatasetInput
from llama_stack.apis.eval import Eval
from llama_stack.apis.inference import Inference
from llama_stack.apis.models import Model, ModelInput
from llama_stack.apis.resource import Resource
from llama_stack.apis.safety import Safety
from llama_stack.apis.scoring import Scoring
from llama_stack.apis.scoring_functions import ScoringFn, ScoringFnInput
from llama_stack.apis.shields import Shield, ShieldInput
from llama_stack.apis.tools import Tool, ToolGroup, ToolGroupInput, ToolRuntime
from llama_stack.apis.vector_dbs import VectorDB, VectorDBInput
from llama_stack.apis.vector_io import VectorIO
from llama_stack.distribution.access_control.datatypes import AccessRule
from llama_stack.providers.datatypes import Api, ProviderSpec
from llama_stack.providers.utils.kvstore.config import KVStoreConfig, SqliteKVStoreConfig
from llama_stack.providers.utils.sqlstore.sqlstore import SqlStoreConfig

LLAMA_STACK_BUILD_CONFIG_VERSION = "2"
LLAMA_STACK_RUN_CONFIG_VERSION = "2"


RoutingKey = str | list[str]


class User(BaseModel):
    principal: str
    # further attributes that may be used for access control decisions
    attributes: dict[str, list[str]] | None = None

    def __init__(self, principal: str, attributes: dict[str, list[str]] | None):
        super().__init__(principal=principal, attributes=attributes)


class ResourceWithOwner(Resource):
    """Extension of Resource that adds an optional owner, i.e. the user that created the
    resource. This can be used to constrain access to the resource."""

    owner: User | None = None


# Use the extended Resource for all routable objects
class ModelWithOwner(Model, ResourceWithOwner):
    pass


class ShieldWithOwner(Shield, ResourceWithOwner):
    pass


class VectorDBWithOwner(VectorDB, ResourceWithOwner):
    pass


class DatasetWithOwner(Dataset, ResourceWithOwner):
    pass


class ScoringFnWithOwner(ScoringFn, ResourceWithOwner):
    pass


class BenchmarkWithOwner(Benchmark, ResourceWithOwner):
    pass


class ToolWithOwner(Tool, ResourceWithOwner):
    pass


class ToolGroupWithOwner(ToolGroup, ResourceWithOwner):
    pass


RoutableObject = Model | Shield | VectorDB | Dataset | ScoringFn | Benchmark | Tool | ToolGroup

RoutableObjectWithProvider = Annotated[
    ModelWithOwner
    | ShieldWithOwner
    | VectorDBWithOwner
    | DatasetWithOwner
    | ScoringFnWithOwner
    | BenchmarkWithOwner
    | ToolWithOwner
    | ToolGroupWithOwner,
    Field(discriminator="type"),
]

RoutedProtocol = Inference | Safety | VectorIO | DatasetIO | Scoring | Eval | ToolRuntime


# Example: /inference, /safety
class AutoRoutedProviderSpec(ProviderSpec):
    provider_type: str = "router"
    config_class: str = ""

    container_image: str | None = None
    routing_table_api: Api
    module: str
    provider_data_validator: str | None = Field(
        default=None,
    )

    @property
    def pip_packages(self) -> list[str]:
        raise AssertionError("Should not be called on AutoRoutedProviderSpec")


# Example: /models, /shields
class RoutingTableProviderSpec(ProviderSpec):
    provider_type: str = "routing_table"
    config_class: str = ""
    container_image: str | None = None

    router_api: Api
    module: str
    pip_packages: list[str] = Field(default_factory=list)


class DistributionSpec(BaseModel):
    description: str | None = Field(
        default="",
        description="Description of the distribution",
    )
    container_image: str | None = None
    providers: dict[str, str | list[str]] = Field(
        default_factory=dict,
        description="""
Provider Types for each of the APIs provided by this distribution. If you
select multiple providers, you should provide an appropriate 'routing_map'
in the runtime configuration to help route to the correct provider.""",
    )


class Provider(BaseModel):
    provider_id: str
    provider_type: str
    config: dict[str, Any]


class LoggingConfig(BaseModel):
    category_levels: dict[str, str] = Field(
        default_factory=dict,
        description="""
 Dictionary of different logging configurations for different portions (ex: core, server) of llama stack""",
    )


class AuthProviderType(str, Enum):
    """Supported authentication provider types."""

    OAUTH2_TOKEN = "oauth2_token"
    CUSTOM = "custom"


class AuthenticationConfig(BaseModel):
    provider_type: AuthProviderType = Field(
        ...,
        description="Type of authentication provider",
    )
    config: dict[str, Any] = Field(
        ...,
        description="Provider-specific configuration",
    )
    access_policy: list[AccessRule] = Field(default=[], description="Rules for determining access to resources")


class AuthenticationRequiredError(Exception):
    pass


class QuotaPeriod(str, Enum):
    DAY = "day"


class QuotaConfig(BaseModel):
    kvstore: SqliteKVStoreConfig = Field(description="Config for KV store backend (SQLite only for now)")
    anonymous_max_requests: int = Field(default=100, description="Max requests for unauthenticated clients per period")
    authenticated_max_requests: int = Field(
        default=1000, description="Max requests for authenticated clients per period"
    )
    period: QuotaPeriod = Field(default=QuotaPeriod.DAY, description="Quota period to set")


class ServerConfig(BaseModel):
    port: int = Field(
        default=8321,
        description="Port to listen on",
        ge=1024,
        le=65535,
    )
    tls_certfile: str | None = Field(
        default=None,
        description="Path to TLS certificate file for HTTPS",
    )
    tls_keyfile: str | None = Field(
        default=None,
        description="Path to TLS key file for HTTPS",
    )
    tls_cafile: str | None = Field(
        default=None,
        description="Path to TLS CA file for HTTPS with mutual TLS authentication",
    )
    auth: AuthenticationConfig | None = Field(
        default=None,
        description="Authentication configuration for the server",
    )
    host: str | None = Field(
        default=None,
        description="The host the server should listen on",
    )
    quota: QuotaConfig | None = Field(
        default=None,
        description="Per client quota request configuration",
    )


class StackRunConfig(BaseModel):
    version: str = LLAMA_STACK_RUN_CONFIG_VERSION

    image_name: str = Field(
        ...,
        description="""
Reference to the distribution this package refers to. For unregistered (adhoc) packages,
this could be just a hash
""",
    )
    container_image: str | None = Field(
        default=None,
        description="Reference to the container image if this package refers to a container",
    )
    apis: list[str] = Field(
        default_factory=list,
        description="""
The list of APIs to serve. If not specified, all APIs specified in the provider_map will be served""",
    )

    providers: dict[str, list[Provider]] = Field(
        description="""
One or more providers to use for each API. The same provider_type (e.g., meta-reference)
can be instantiated multiple times (with different configs) if necessary.
""",
    )
    metadata_store: KVStoreConfig | None = Field(
        default=None,
        description="""
Configuration for the persistence store used by the distribution registry. If not specified,
a default SQLite store will be used.""",
    )

    inference_store: SqlStoreConfig | None = Field(
        default=None,
        description="""
Configuration for the persistence store used by the inference API. If not specified,
a default SQLite store will be used.""",
    )

    # registry of "resources" in the distribution
    models: list[ModelInput] = Field(default_factory=list)
    shields: list[ShieldInput] = Field(default_factory=list)
    vector_dbs: list[VectorDBInput] = Field(default_factory=list)
    datasets: list[DatasetInput] = Field(default_factory=list)
    scoring_fns: list[ScoringFnInput] = Field(default_factory=list)
    benchmarks: list[BenchmarkInput] = Field(default_factory=list)
    tool_groups: list[ToolGroupInput] = Field(default_factory=list)

    logging: LoggingConfig | None = Field(default=None, description="Configuration for Llama Stack Logging")

    server: ServerConfig = Field(
        default_factory=ServerConfig,
        description="Configuration for the HTTP(S) server",
    )

    external_providers_dir: Path | None = Field(
        default=None,
        description="Path to directory containing external provider implementations. The providers code and dependencies must be installed on the system.",
    )

    @field_validator("external_providers_dir")
    @classmethod
    def validate_external_providers_dir(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v)
        return v


class BuildConfig(BaseModel):
    version: str = LLAMA_STACK_BUILD_CONFIG_VERSION

    distribution_spec: DistributionSpec = Field(description="The distribution spec to build including API providers. ")
    image_type: str = Field(
        default="conda",
        description="Type of package to build (conda | container | venv)",
    )
    image_name: str | None = Field(
        default=None,
        description="Name of the distribution to build",
    )
    external_providers_dir: Path | None = Field(
        default=None,
        description="Path to directory containing external provider implementations. The providers packages will be resolved from this directory. "
        "pip_packages MUST contain the provider package name.",
    )
    additional_pip_packages: list[str] = Field(
        default_factory=list,
        description="Additional pip packages to install in the distribution. These packages will be installed in the distribution environment.",
    )

    @field_validator("external_providers_dir")
    @classmethod
    def validate_external_providers_dir(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v)
        return v
