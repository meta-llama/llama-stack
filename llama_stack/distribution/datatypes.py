# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

from llama_stack.providers.datatypes import *  # noqa: F403
from llama_stack.apis.models import *  # noqa: F403
from llama_stack.apis.shields import *  # noqa: F403
from llama_stack.apis.memory_banks import *  # noqa: F403
from llama_stack.apis.datasets import *  # noqa: F403
from llama_stack.apis.scoring_functions import *  # noqa: F403
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.inference import Inference
from llama_stack.apis.memory import Memory
from llama_stack.apis.safety import Safety
from llama_stack.apis.scoring import Scoring
from llama_stack.providers.utils.kvstore.config import KVStoreConfig

LLAMA_STACK_BUILD_CONFIG_VERSION = "2"
LLAMA_STACK_RUN_CONFIG_VERSION = "2"


RoutingKey = Union[str, List[str]]


RoutableObject = Union[
    ModelDef,
    ShieldDef,
    MemoryBankDef,
    DatasetDef,
    ScoringFnDef,
]


RoutableObjectWithProvider = Annotated[
    Union[
        ModelDefWithProvider,
        ShieldDefWithProvider,
        MemoryBankDefWithProvider,
        DatasetDefWithProvider,
        ScoringFnDefWithProvider,
    ],
    Field(discriminator="type"),
]

RoutedProtocol = Union[
    Inference,
    Safety,
    Memory,
    DatasetIO,
    Scoring,
]


# Example: /inference, /safety
class AutoRoutedProviderSpec(ProviderSpec):
    provider_type: str = "router"
    config_class: str = ""

    docker_image: Optional[str] = None
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
    docker_image: Optional[str] = None

    router_api: Api
    module: str
    pip_packages: List[str] = Field(default_factory=list)


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


class Provider(BaseModel):
    provider_id: str
    provider_type: str
    config: Dict[str, Any]


class StackRunConfig(BaseModel):
    version: str = LLAMA_STACK_RUN_CONFIG_VERSION
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


class BuildConfig(BaseModel):
    version: str = LLAMA_STACK_BUILD_CONFIG_VERSION
    name: str
    distribution_spec: DistributionSpec = Field(
        description="The distribution spec to build including API providers. "
    )
    image_type: str = Field(
        default="conda",
        description="Type of package to build (conda | container)",
    )
