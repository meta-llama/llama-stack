# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

from llama_models.schema_utils import json_schema_type, webmethod
from pydantic import Field

from llama_stack.apis.resource import Resource, ResourceType


@json_schema_type
class Model(Resource):
    type: Literal[ResourceType.model.value] = ResourceType.model.value
    llama_model: str = Field(
        description="Pointer to the underlying core Llama family model. Each model served by Llama Stack must have a core Llama model.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata for this model",
    )


@runtime_checkable
class Models(Protocol):
    @webmethod(route="/models/list", method="GET")
    async def list_models(self) -> List[Model]: ...

    @webmethod(route="/models/get", method="GET")
    async def get_model(self, identifier: str) -> Optional[Model]: ...

    @webmethod(route="/models/register", method="POST")
    async def register_model(
        self,
        model_id: str,
        provider_model_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        llama_model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Model: ...
