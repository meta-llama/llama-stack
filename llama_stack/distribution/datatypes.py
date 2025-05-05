# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field

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
from llama_stack.providers.datatypes import Api, ProviderSpec
from llama_stack.providers.utils.kvstore.config import KVStoreConfig

LLAMA_STACK_BUILD_CONFIG_VERSION = "2"
LLAMA_STACK_RUN_CONFIG_VERSION = "2"


RoutingKey = str | list[str]


class AccessAttributes(BaseModel):
    """Structured representation of user attributes for access control.

    This model defines a structured approach to representing user attributes
    with common standard categories for access control.

    Standard attribute categories include:
    - roles: Role-based attributes (e.g., admin, data-scientist)
    - teams: Team-based attributes (e.g., ml-team, infra-team)
    - projects: Project access attributes (e.g., llama-3, customer-insights)
    - namespaces: Namespace-based access control for resource isolation
    """

    # Standard attribute categories - the minimal set we need now
    roles: list[str] | None = Field(
        default=None, description="Role-based attributes (e.g., 'admin', 'data-scientist', 'user')"
    )

    teams: list[str] | None = Field(default=None, description="Team-based attributes (e.g., 'ml-team', 'nlp-team')")

    projects: list[str] | None = Field(
        default=None, description="Project-based access attributes (e.g., 'llama-3', 'customer-insights')"
    )

    namespaces: list[str] | None = Field(
        default=None, description="Namespace-based access control for resource isolation"
    )


class ResourceWithACL(Resource):
    """Extension of Resource that adds attribute-based access control capabilities.

    This class adds an optional access_attributes field that allows fine-grained control
    over which users can access each resource. When attributes are defined, a user must have
    matching attributes to access the resource.

    Attribute Matching Algorithm:
    1. If a resource has no access_attributes (None or empty dict), it's visible to all authenticated users
    2. Each key in access_attributes represents an attribute category (e.g., "roles", "teams", "projects")
    3. The matching algorithm requires ALL categories to match (AND relationship between categories)
    4. Within each category, ANY value match is sufficient (OR relationship within a category)

    Examples:
        # Resource visible to everyone (no access control)
        model = Model(identifier="llama-2", ...)

        # Resource visible only to admins
        model = Model(
            identifier="gpt-4",
            access_attributes=AccessAttributes(roles=["admin"])
        )

        # Resource visible to data scientists on the ML team
        model = Model(
            identifier="private-model",
            access_attributes=AccessAttributes(
                roles=["data-scientist", "researcher"],
                teams=["ml-team"]
            )
        )
        # ^ User must have at least one of the roles AND be on the ml-team

        # Resource visible to users with specific project access
        vector_db = VectorDB(
            identifier="customer-embeddings",
            access_attributes=AccessAttributes(
                projects=["customer-insights"],
                namespaces=["confidential"]
            )
        )
        # ^ User must have access to the customer-insights project AND have confidential namespace
    """

    access_attributes: AccessAttributes | None = None


# Use the extended Resource for all routable objects
class ModelWithACL(Model, ResourceWithACL):
    pass


class ShieldWithACL(Shield, ResourceWithACL):
    pass


class VectorDBWithACL(VectorDB, ResourceWithACL):
    pass


class DatasetWithACL(Dataset, ResourceWithACL):
    pass


class ScoringFnWithACL(ScoringFn, ResourceWithACL):
    pass


class BenchmarkWithACL(Benchmark, ResourceWithACL):
    pass


class ToolWithACL(Tool, ResourceWithACL):
    pass


class ToolGroupWithACL(ToolGroup, ResourceWithACL):
    pass


RoutableObject = Model | Shield | VectorDB | Dataset | ScoringFn | Benchmark | Tool | ToolGroup

RoutableObjectWithProvider = Annotated[
    ModelWithACL
    | ShieldWithACL
    | VectorDBWithACL
    | DatasetWithACL
    | ScoringFnWithACL
    | BenchmarkWithACL
    | ToolWithACL
    | ToolGroupWithACL,
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

    KUBERNETES = "kubernetes"
    CUSTOM = "custom"


class AuthenticationConfig(BaseModel):
    provider_type: AuthProviderType = Field(
        ...,
        description="Type of authentication provider (e.g., 'kubernetes', 'custom')",
    )
    config: dict[str, str] = Field(
        ...,
        description="Provider-specific configuration",
    )


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
    auth: AuthenticationConfig | None = Field(
        default=None,
        description="Authentication configuration for the server",
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

    external_providers_dir: str | None = Field(
        default=None,
        description="Path to directory containing external provider implementations. The providers code and dependencies must be installed on the system.",
    )


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
    external_providers_dir: str | None = Field(
        default=None,
        description="Path to directory containing external provider implementations. The providers packages will be resolved from this directory. "
        "pip_packages MUST contain the provider package name.",
    )
