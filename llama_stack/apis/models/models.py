# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List, Optional, Protocol

from llama_models.schema_utils import json_schema_type, webmethod
from pydantic import BaseModel, Field


@json_schema_type
class ModelDef(BaseModel):
    identifier: str = Field(
        description="A unique identifier for the model type",
    )
    llama_model: str = Field(
        description="Pointer to the core Llama family model",
    )
    provider_id: str = Field(
        description="The provider instance which serves this model"
    )
    # For now, we are only supporting core llama models but as soon as finetuned
    # and other custom models (for example various quantizations) are allowed, there
    # will be more metadata fields here


class Models(Protocol):
    @webmethod(route="/models/list", method="GET")
    async def list_models(self) -> List[ModelDef]: ...

    @webmethod(route="/models/get", method="GET")
    async def get_model(self, identifier: str) -> Optional[ModelDef]: ...

    @webmethod(route="/models/register", method="POST")
    async def register_model(self, model: ModelDef) -> None: ...
