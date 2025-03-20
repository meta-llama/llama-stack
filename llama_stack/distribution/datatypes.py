# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated, Any, Dict, List, Optional, Union

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


RoutingKey = Union[str, List[str]]


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
    roles: Optional[List[str]] = Field(
        default=None, description="Role-based attributes (e.g., 'admin', 'data-scientist', 'user')"
    )

    teams: Optional[List[str]] = Field(default=None, description="Team-based attributes (e.g., 'ml-team', 'nlp-team')")

    projects: Optional[List[str]] = Field(
        default=None, description="Project-based access attributes (e.g., 'llama-3', 'customer-insights')"
    )

    namespaces: Optional[List[str]] = Field(
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

    access_attributes: Optional[AccessAttributes] = None


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


RoutableObject = Union[
    Model,
    Shield,
    VectorDB,
    Dataset,
    ScoringFn,
    Benchmark,
    Tool,
    ToolGroup,
]


RoutableObjectWithProvider = Annotated[
    Union[
        ModelWithACL,
        ShieldWithACL,
        VectorDBWithACL,
        DatasetWithACL,
        ScoringFnWithACL,
        BenchmarkWithACL,
        ToolWithACL,
        ToolGroupWithACL,
    ],
    Field(discriminator="type"),
]

RoutedProtocol = Union[
    Inference,
    Safety,
    VectorIO,
    DatasetIO,
    Scoring,
    Eval,
    ToolRuntime,
]


# Example: /inference, /safety
class AutoRoutedProviderSpec(ProviderSpec):
    provider_type: str = "router"
    config_class: str = ""

    container_image: Optional[str] = None
    routing_table_api: Api
    module: str
    provider_data_validator: Optional[str] = Field(
        default=None,
    )

    @property
    def pip_packages(self) -> List[str]:
        raise AssertionError("Should not be called on AutoRoutedProviderSpec")


# Example: /models, /shields
class RoutingTableProviderSpec(ProviderSpec):
    provider_type: str = "routing_table"
    config_class: str = ""
    container_image: Optional[str] = None

    router_api: Api
    module: str
    pip_packages: List[str] = Field(default_factory=list)


class DistributionSpec(BaseModel):
    description: Optional[str] = Field(
        default="",
        description="Description of the distribution",
    )
    container_image: Optional[str] = None
    providers: Dict[str, Union[str, List[str]]] = Field(
        default_factory=dict,
        description="""
Provider Types for each of the APIs provided by this distribution. If you
select multiple providers, you should provide an appropriate 'routing_map'
in the runtime configuration to help route to the correct provider.""",
    )


class Provider(BaseModel):
    provider_id: str
    provider_type: str
    config: Dict[str, Any]


class LoggingConfig(BaseModel):
    category_levels: Dict[str, str] = Field(
        default_factory=Dict,
        description="""
 Dictionary of different logging configurations for different portions (ex: core, server) of llama stack""",
    )


class AuthenticationConfig(BaseModel):
    endpoint: str = Field(
        ...,
        description="Endpoint URL to validate authentication tokens",
    )


class ServerConfig(BaseModel):
    port: int = Field(
        default=8321,
        description="Port to listen on",
        ge=1024,
        le=65535,
    )
    tls_certfile: Optional[str] = Field(
        default=None,
        description="Path to TLS certificate file for HTTPS",
    )
    tls_keyfile: Optional[str] = Field(
        default=None,
        description="Path to TLS key file for HTTPS",
    )
    auth: Optional[AuthenticationConfig] = Field(
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
    container_image: Optional[str] = Field(
        default=None,
        description="Reference to the container image if this package refers to a container",
    )
    apis: List[str] = Field(
        default_factory=list,
        description="""
The list of APIs to serve. If not specified, all APIs specified in the provider_map will be served""",
    )

    providers: Dict[str, List[Provider]] = Field(
        description="""
One or more providers to use for each API. The same provider_type (e.g., meta-reference)
can be instantiated multiple times (with different configs) if necessary.
""",
    )
    metadata_store: Optional[KVStoreConfig] = Field(
        default=None,
        description="""
Configuration for the persistence store used by the distribution registry. If not specified,
a default SQLite store will be used.""",
    )

    # registry of "resources" in the distribution
    models: List[ModelInput] = Field(default_factory=list)
    shields: List[ShieldInput] = Field(default_factory=list)
    vector_dbs: List[VectorDBInput] = Field(default_factory=list)
    datasets: List[DatasetInput] = Field(default_factory=list)
    scoring_fns: List[ScoringFnInput] = Field(default_factory=list)
    benchmarks: List[BenchmarkInput] = Field(default_factory=list)
    tool_groups: List[ToolGroupInput] = Field(default_factory=list)

    logging: Optional[LoggingConfig] = Field(default=None, description="Configuration for Llama Stack Logging")

    server: ServerConfig = Field(
        default_factory=ServerConfig,
        description="Configuration for the HTTP(S) server",
    )


class BuildConfig(BaseModel):
    version: str = LLAMA_STACK_BUILD_CONFIG_VERSION

    distribution_spec: DistributionSpec = Field(description="The distribution spec to build including API providers. ")
    image_type: str = Field(
        default="conda",
        description="Type of package to build (conda | container | venv)",
    )
