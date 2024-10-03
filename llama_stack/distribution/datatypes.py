# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

from llama_stack.providers.datatypes import *  # noqa: F403


LLAMA_STACK_BUILD_CONFIG_VERSION = "v1"
LLAMA_STACK_RUN_CONFIG_VERSION = "v1"


RoutingKey = Union[str, List[str]]


class GenericProviderConfig(BaseModel):
    provider_type: str
    config: Dict[str, Any]


class RoutableProviderConfig(GenericProviderConfig):
    routing_key: RoutingKey


class PlaceholderProviderConfig(BaseModel):
    """Placeholder provider config for API whose provider are defined in routing_table"""

    providers: List[str]


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
@json_schema_type
class RoutingTableProviderSpec(ProviderSpec):
    provider_type: str = "routing_table"
    config_class: str = ""
    docker_image: Optional[str] = None

    inner_specs: List[ProviderSpec]
    module: str
    pip_packages: List[str] = Field(default_factory=list)


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
    apis_to_serve: List[str] = Field(
        description="""
The list of APIs to serve. If not specified, all APIs specified in the provider_map will be served""",
    )

    api_providers: Dict[
        str, Union[GenericProviderConfig, PlaceholderProviderConfig]
    ] = Field(
        description="""
Provider configurations for each of the APIs provided by this package.
""",
    )
    routing_table: Dict[str, List[RoutableProviderConfig]] = Field(
        default_factory=dict,
        description="""

        E.g. The following is a ProviderRoutingEntry for models:
        - routing_key: Llama3.1-8B-Instruct
          provider_type: meta-reference
          config:
              model: Llama3.1-8B-Instruct
              quantization: null
              torch_seed: null
              max_seq_len: 4096
              max_batch_size: 1
        """,
    )


@json_schema_type
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
