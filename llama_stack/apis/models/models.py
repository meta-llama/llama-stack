# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict, List, Optional, Protocol

from llama_models.llama3.api.datatypes import *  # noqa: F403

from llama_models.schema_utils import json_schema_type, webmethod
from pydantic import BaseModel, Field


@json_schema_type
class ModelSpec(BaseModel):
    llama_model_metadata: Model = Field(
        description="All metadatas associated with llama model (defined in llama_models.models.sku_list). "
    )
    providers_spec: Dict[str, List[Any]] = Field(
        default_factory=dict,
        description="Map of API to the concrete provider specs. E.g. {}".format(
            {
                "inference": [
                    {
                        "provider_type": "remote::8080",
                        "url": "localhost::5555",
                        "api_token": "hf_xxx",
                    },
                    {
                        "provider_type": "meta-reference",
                        "model": "Meta-Llama3.1-8B-Instruct",
                        "max_seq_len": 4096,
                    },
                ]
            }
        ),
    )


@json_schema_type
class ModelsListResponse(BaseModel):
    models_list: List[ModelSpec]


@json_schema_type
class ModelsGetResponse(BaseModel):
    core_model_spec: Optional[ModelSpec] = None


@json_schema_type
class ModelsRegisterResponse(BaseModel):
    core_model_spec: Optional[ModelSpec] = None


class Models(Protocol):
    @webmethod(route="/models/list", method="GET")
    async def list_models(self) -> ModelsListResponse: ...

    @webmethod(route="/models/get", method="POST")
    async def get_model(self, model_id: str) -> ModelsGetResponse: ...

    @webmethod(route="/models/register")
    async def register_model(
        self, model_id: str, api: str, provider_spec: Dict[str, str]
    ) -> ModelsRegisterResponse: ...
