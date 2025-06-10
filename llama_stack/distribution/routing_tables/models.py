# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time
from typing import Any

from llama_stack.apis.models import ListModelsResponse, Model, Models, ModelType, OpenAIListModelsResponse, OpenAIModel
from llama_stack.distribution.datatypes import (
    ModelWithOwner,
)
from llama_stack.log import get_logger

from .common import CommonRoutingTableImpl

logger = get_logger(name=__name__, category="core")


class ModelsRoutingTable(CommonRoutingTableImpl, Models):
    async def list_models(self) -> ListModelsResponse:
        return ListModelsResponse(data=await self.get_all_with_type("model"))

    async def openai_list_models(self) -> OpenAIListModelsResponse:
        models = await self.get_all_with_type("model")
        openai_models = [
            OpenAIModel(
                id=model.identifier,
                object="model",
                created=int(time.time()),
                owned_by="llama_stack",
            )
            for model in models
        ]
        return OpenAIListModelsResponse(data=openai_models)

    async def get_model(self, model_id: str) -> Model:
        model = await self.get_object_by_identifier("model", model_id)
        if model is None:
            raise ValueError(f"Model '{model_id}' not found")
        return model

    async def register_model(
        self,
        model_id: str,
        provider_model_id: str | None = None,
        provider_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        model_type: ModelType | None = None,
    ) -> Model:
        if provider_model_id is None:
            provider_model_id = model_id
        if provider_id is None:
            # If provider_id not specified, use the only provider if it supports this model
            if len(self.impls_by_provider_id) == 1:
                provider_id = list(self.impls_by_provider_id.keys())[0]
            else:
                raise ValueError(
                    f"No provider specified and multiple providers available. Please specify a provider_id. Available providers: {self.impls_by_provider_id.keys()}"
                )
        if metadata is None:
            metadata = {}
        if model_type is None:
            model_type = ModelType.llm
        if "embedding_dimension" not in metadata and model_type == ModelType.embedding:
            raise ValueError("Embedding model must have an embedding dimension in its metadata")
        model = ModelWithOwner(
            identifier=model_id,
            provider_resource_id=provider_model_id,
            provider_id=provider_id,
            metadata=metadata,
            model_type=model_type,
        )
        registered_model = await self.register_object(model)
        return registered_model

    async def unregister_model(self, model_id: str) -> None:
        existing_model = await self.get_model(model_id)
        if existing_model is None:
            raise ValueError(f"Model {model_id} not found")
        await self.unregister_object(existing_model)
