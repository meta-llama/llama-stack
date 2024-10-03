# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List, Optional, Protocol

from llama_models.llama3.api.datatypes import Model

from llama_models.schema_utils import json_schema_type, webmethod
from pydantic import BaseModel, Field

from llama_stack.distribution.datatypes import GenericProviderConfig


@json_schema_type
class ModelServingSpec(BaseModel):
    llama_model: Model = Field(
        description="All metadatas associated with llama model (defined in llama_models.models.sku_list).",
    )
    provider_config: GenericProviderConfig = Field(
        description="Provider config for the model, including provider_type, and corresponding config. ",
    )


class Models(Protocol):
    @webmethod(route="/models/list", method="GET")
    async def list_models(self) -> List[ModelServingSpec]: ...

    @webmethod(route="/models/get", method="GET")
    async def get_model(self, core_model_id: str) -> Optional[ModelServingSpec]: ...
