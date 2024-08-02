# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, Field
from strong_typing.schema import json_schema_type


@json_schema_type
class ApiSurface(Enum):
    inference = "inference"
    safety = "safety"


@json_schema_type
class ApiSurfaceEndpoint(BaseModel):
    route: str
    method: str
    name: str


@json_schema_type
class Adapter(BaseModel):
    api_surface: ApiSurface
    adapter_id: str


@json_schema_type
class SourceAdapter(Adapter):
    pip_packages: List[str] = Field(
        default_factory=list,
        description="The pip dependencies needed for this implementation",
    )
    module: str = Field(
        ...,
        description="""
Fully-qualified name of the module to import. The module is expected to have
a `get_adapter_instance()` method which will be passed a validated config object
of type `config_class`.""",
    )
    config_class: str = Field(
        ...,
        description="Fully-qualified classname of the config for this adapter",
    )


@json_schema_type
class PassthroughApiAdapter(Adapter):
    base_url: str = Field(..., description="The base URL for the llama stack provider")
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Headers (e.g., authorization) to send with the request",
    )


class Distribution(BaseModel):
    name: str
    description: str

    adapters: Dict[ApiSurface, Adapter] = Field(
        default_factory=dict,
        description="The API surfaces provided by this distribution",
    )

    additional_pip_packages: List[str] = Field(
        default_factory=list,
        description="Additional pip packages beyond those required by the adapters",
    )
