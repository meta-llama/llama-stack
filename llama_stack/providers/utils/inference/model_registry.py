# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections import namedtuple
from typing import List

from llama_stack.providers.datatypes import Model, ModelsProtocolPrivate

ModelAlias = namedtuple("ModelAlias", ["provider_model_id", "aliases", "llama_model"])


class ModelLookup:
    def __init__(
        self,
        model_aliases: List[ModelAlias],
    ):
        self.alias_to_provider_id_map = {}
        self.provider_id_to_llama_model_map = {}
        for alias_obj in model_aliases:
            for alias in alias_obj.aliases:
                self.alias_to_provider_id_map[alias] = alias_obj.provider_model_id
            # also add a mapping from provider model id to itself for easy lookup
            self.alias_to_provider_id_map[alias_obj.provider_model_id] = (
                alias_obj.provider_model_id
            )
            self.provider_id_to_llama_model_map[alias_obj.provider_model_id] = (
                alias_obj.llama_model
            )

    def get_provider_model_id(self, identifier: str) -> str:
        if identifier in self.alias_to_provider_id_map:
            return self.alias_to_provider_id_map[identifier]
        else:
            raise ValueError(f"Unknown model: `{identifier}`")


class ModelRegistryHelper(ModelsProtocolPrivate):

    def __init__(self, model_aliases: List[ModelAlias]):
        self.model_lookup = ModelLookup(model_aliases)

    def get_llama_model(self, provider_model_id: str) -> str:
        return self.model_lookup.provider_id_to_llama_model_map[provider_model_id]

    async def register_model(self, model: Model) -> Model:
        provider_model_id = self.model_lookup.get_provider_model_id(
            model.provider_resource_id
        )
        if not provider_model_id:
            raise ValueError(f"Unknown model: `{model.provider_resource_id}`")

        model.provider_resource_id = provider_model_id

        return model
