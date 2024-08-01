# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel, Field
from strong_typing.schema import json_schema_type
from typing_extensions import Annotated


@json_schema_type
class AdapterType(Enum):
    passthrough_api = "passthrough_api"
    python_impl = "python_impl"
    not_implemented = "not_implemented"


@json_schema_type
class PassthroughApiAdapterConfig(BaseModel):
    type: Literal[AdapterType.passthrough_api.value] = AdapterType.passthrough_api.value
    base_url: str = Field(..., description="The base URL for the llama stack provider")
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Headers (e.g., authorization) to send with the request",
    )


@json_schema_type
class PythonImplAdapterConfig(BaseModel):
    type: Literal[AdapterType.python_impl.value] = AdapterType.python_impl.value
    pip_packages: List[str] = Field(
        default_factory=list,
        description="The pip dependencies needed for this implementation",
    )
    module: str = Field(..., description="The name of the module to import")
    entrypoint: str = Field(
        ...,
        description="The name of the entrypoint function which creates the implementation for the API",
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="kwargs to pass to the entrypoint"
    )


@json_schema_type
class NotImplementedAdapterConfig(BaseModel):
    type: Literal[AdapterType.not_implemented.value] = AdapterType.not_implemented.value


# should we define very granular typed classes for each of the PythonImplAdapters we will have?
# e.g., OllamaInference / vLLMInference / etc. might need very specific parameters
AdapterConfig = Annotated[
    Union[
        PassthroughApiAdapterConfig,
        NotImplementedAdapterConfig,
        PythonImplAdapterConfig,
    ],
    Field(discriminator="type"),
]


class DistributionConfig(BaseModel):
    inference: AdapterConfig
    safety: AdapterConfig

    # configs for each API that the stack provides, e.g.
    # agentic_system: AdapterConfig
    # post_training: AdapterConfig


class DistributionConfigDefaults(BaseModel):
    inference: Dict[str, Any] = Field(
        default_factory=dict, description="Default kwargs for the inference adapter"
    )
    safety: Dict[str, Any] = Field(
        default_factory=dict, description="Default kwargs for the safety adapter"
    )


class Distribution(BaseModel):
    name: str
    description: str

    # you must install the packages to get the functionality needed.
    # later, we may have a docker image be the main artifact of
    # a distribution.
    pip_packages: List[str]

    config_defaults: DistributionConfigDefaults
