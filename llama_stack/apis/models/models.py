# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

from llama_models.schema_utils import json_schema_type, webmethod
from pydantic import BaseModel, Field


class ModelDef(BaseModel):
    identifier: str = Field(
        description="A unique name for the model type",
    )
    llama_model: str = Field(
        description="Pointer to the underlying core Llama family model. Each model served by Llama Stack must have a core Llama model.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata for this model",
    )


@json_schema_type
class ModelDefWithProvider(ModelDef):
    type: Literal["model"] = "model"
    provider_id: str = Field(
        description="The provider ID for this model",
    )


@runtime_checkable
class Models(Protocol):
    @webmethod(route="/models/list", method="GET")
    async def list_models(self) -> List[ModelDefWithProvider]: ...

    @webmethod(route="/models/get", method="GET")
    async def get_model(self, identifier: str) -> Optional[ModelDefWithProvider]: ...

    @webmethod(route="/models/register", method="POST")
    async def register_model(self, model: ModelDefWithProvider) -> None: ...
