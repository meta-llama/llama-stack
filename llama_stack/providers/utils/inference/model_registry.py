# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.apis.models.models import ModelType
from llama_stack.models.llama.sku_list import all_registered_models
from llama_stack.providers.datatypes import Model, ModelsProtocolPrivate
from llama_stack.providers.utils.inference import (
    ALL_HUGGINGFACE_REPOS_TO_MODEL_DESCRIPTOR,
)


# TODO: this class is more confusing than useful right now. We need to make it
# more closer to the Model class.
class ProviderModelEntry(BaseModel):
    provider_model_id: str
    aliases: list[str] = Field(default_factory=list)
    llama_model: str | None = None
    model_type: ModelType = ModelType.llm
    metadata: dict[str, Any] = Field(default_factory=dict)


def get_huggingface_repo(model_descriptor: str) -> str | None:
    for model in all_registered_models():
        if model.descriptor() == model_descriptor:
            return model.huggingface_repo
    return None


def build_hf_repo_model_entry(
    provider_model_id: str, model_descriptor: str, additional_aliases: list[str] | None = None
) -> ProviderModelEntry:
    aliases = [
        get_huggingface_repo(model_descriptor),
    ]
    if additional_aliases:
        aliases.extend(additional_aliases)
    return ProviderModelEntry(
        provider_model_id=provider_model_id,
        aliases=aliases,
        llama_model=model_descriptor,
    )


def build_model_entry(provider_model_id: str, model_descriptor: str) -> ProviderModelEntry:
    return ProviderModelEntry(
        provider_model_id=provider_model_id,
        aliases=[],
        llama_model=model_descriptor,
        model_type=ModelType.llm,
    )


class ModelRegistryHelper(ModelsProtocolPrivate):
    def __init__(self, model_entries: list[ProviderModelEntry]):
        self.alias_to_provider_id_map = {}
        self.provider_id_to_llama_model_map = {}
        for entry in model_entries:
            for alias in entry.aliases:
                self.alias_to_provider_id_map[alias] = entry.provider_model_id

            # also add a mapping from provider model id to itself for easy lookup
            self.alias_to_provider_id_map[entry.provider_model_id] = entry.provider_model_id

            if entry.llama_model:
                self.alias_to_provider_id_map[entry.llama_model] = entry.provider_model_id
                self.provider_id_to_llama_model_map[entry.provider_model_id] = entry.llama_model

    def get_provider_model_id(self, identifier: str) -> str | None:
        return self.alias_to_provider_id_map.get(identifier, None)

    # TODO: why keep a separate llama model mapping?
    def get_llama_model(self, provider_model_id: str) -> str | None:
        return self.provider_id_to_llama_model_map.get(provider_model_id, None)

    async def register_model(self, model: Model) -> Model:
        if not (supported_model_id := self.get_provider_model_id(model.provider_resource_id)):
            raise ValueError(
                f"Model '{model.provider_resource_id}' is not supported. Supported models are: {', '.join(self.alias_to_provider_id_map.keys())}"
            )
        provider_resource_id = self.get_provider_model_id(model.model_id)
        if model.model_type == ModelType.embedding:
            # embedding models are always registered by their provider model id and does not need to be mapped to a llama model
            provider_resource_id = model.provider_resource_id
        if provider_resource_id:
            if provider_resource_id != supported_model_id:  # be idemopotent, only reject differences
                raise ValueError(
                    f"Model id '{model.model_id}' is already registered. Please use a different id or unregister it first."
                )
        else:
            llama_model = model.metadata.get("llama_model")
            if llama_model:
                existing_llama_model = self.get_llama_model(model.provider_resource_id)
                if existing_llama_model:
                    if existing_llama_model != llama_model:
                        raise ValueError(
                            f"Provider model id '{model.provider_resource_id}' is already registered to a different llama model: '{existing_llama_model}'"
                        )
                else:
                    if llama_model not in ALL_HUGGINGFACE_REPOS_TO_MODEL_DESCRIPTOR:
                        raise ValueError(
                            f"Invalid llama_model '{llama_model}' specified in metadata. "
                            f"Must be one of: {', '.join(ALL_HUGGINGFACE_REPOS_TO_MODEL_DESCRIPTOR.keys())}"
                        )
                    self.provider_id_to_llama_model_map[model.provider_resource_id] = (
                        ALL_HUGGINGFACE_REPOS_TO_MODEL_DESCRIPTOR[llama_model]
                    )

        self.alias_to_provider_id_map[model.model_id] = supported_model_id

        return model

    async def unregister_model(self, model_id: str) -> None:
        # TODO: should we block unregistering base supported provider model IDs?
        if model_id not in self.alias_to_provider_id_map:
            raise ValueError(f"Model id '{model_id}' is not registered.")

        del self.alias_to_provider_id_map[model_id]
