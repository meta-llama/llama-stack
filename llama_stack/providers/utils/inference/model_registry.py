# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections import namedtuple
from typing import List, Optional

from llama_models.sku_list import all_registered_models

from llama_stack.providers.datatypes import Model, ModelsProtocolPrivate

from llama_stack.providers.utils.inference import (
    ALL_HUGGINGFACE_REPOS_TO_MODEL_DESCRIPTOR,
)

ModelAlias = namedtuple("ModelAlias", ["provider_model_id", "aliases", "llama_model"])


def get_huggingface_repo(model_descriptor: str) -> Optional[str]:
    for model in all_registered_models():
        if model.descriptor() == model_descriptor:
            return model.huggingface_repo
    return None


def build_model_alias(provider_model_id: str, model_descriptor: str) -> ModelAlias:
    return ModelAlias(
        provider_model_id=provider_model_id,
        aliases=[
            get_huggingface_repo(model_descriptor),
        ],
        llama_model=model_descriptor,
    )


def build_model_alias_with_just_provider_model_id(
    provider_model_id: str, model_descriptor: str
) -> ModelAlias:
    return ModelAlias(
        provider_model_id=provider_model_id,
        aliases=[],
        llama_model=model_descriptor,
    )


class ModelRegistryHelper(ModelsProtocolPrivate):
    def __init__(self, model_aliases: List[ModelAlias]):
        self.alias_to_provider_id_map = {}
        self.provider_id_to_llama_model_map = {}
        for alias_obj in model_aliases:
            for alias in alias_obj.aliases:
                self.alias_to_provider_id_map[alias] = alias_obj.provider_model_id
            # also add a mapping from provider model id to itself for easy lookup
            self.alias_to_provider_id_map[alias_obj.provider_model_id] = (
                alias_obj.provider_model_id
            )
            # ensure we can go from llama model to provider model id
            self.alias_to_provider_id_map[alias_obj.llama_model] = (
                alias_obj.provider_model_id
            )
            self.provider_id_to_llama_model_map[alias_obj.provider_model_id] = (
                alias_obj.llama_model
            )

    def get_provider_model_id(self, identifier: str) -> str:
        if identifier in self.alias_to_provider_id_map:
            return self.alias_to_provider_id_map[identifier]
        else:
            return None

    def get_llama_model(self, provider_model_id: str) -> str:
        if provider_model_id in self.provider_id_to_llama_model_map:
            return self.provider_id_to_llama_model_map[provider_model_id]
        else:
            return None

    async def register_model(self, model: Model) -> Model:
        provider_resource_id = self.get_provider_model_id(model.provider_resource_id)
        if provider_resource_id:
            model.provider_resource_id = provider_resource_id
        else:
            if model.metadata.get("llama_model") is None:
                raise ValueError(
                    f"Model '{model.provider_resource_id}' is not available and no llama_model was specified in metadata. "
                    "Please specify a llama_model in metadata or use a supported model identifier"
                )
            existing_llama_model = self.get_llama_model(model.provider_resource_id)
            if existing_llama_model:
                if existing_llama_model != model.metadata["llama_model"]:
                    raise ValueError(
                        f"Provider model id '{model.provider_resource_id}' is already registered to a different llama model: '{existing_llama_model}'"
                    )
            else:
                if (
                    model.metadata["llama_model"]
                    not in ALL_HUGGINGFACE_REPOS_TO_MODEL_DESCRIPTOR
                ):
                    raise ValueError(
                        f"Invalid llama_model '{model.metadata['llama_model']}' specified in metadata. "
                        f"Must be one of: {', '.join(ALL_HUGGINGFACE_REPOS_TO_MODEL_DESCRIPTOR.keys())}"
                    )
                self.provider_id_to_llama_model_map[model.provider_resource_id] = (
                    ALL_HUGGINGFACE_REPOS_TO_MODEL_DESCRIPTOR[
                        model.metadata["llama_model"]
                    ]
                )

        return model
