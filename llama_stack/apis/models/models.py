# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Optional, Protocol

from llama_models.llama3.api.datatypes import Model

from llama_models.schema_utils import json_schema_type, webmethod
from llama_stack.distribution.datatypes import GenericProviderConfig
from pydantic import BaseModel, Field


@json_schema_type
class ModelSpec(BaseModel):
    llama_model_metadata: Model = Field(
        description="All metadatas associated with llama model (defined in llama_models.models.sku_list). "
    )
    provider_id: str = Field(
        description="API provider that is serving this model (e.g. meta-reference, local)",
    )
    api: str = Field(
        description="API that this model is serving (e.g. inference / safety)",
    )
    provider_config: Dict[str, Any] = Field(
        description="API provider config used for serving this model to the API provider `provider_id`"
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
    async def get_model(self, core_model_id: str) -> ModelsGetResponse: ...
