# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict, List

from llama_models.sku_list import resolve_model

from llama_stack.apis.models import *  # noqa: F403


class ModelRegistryHelper:

    def __init__(self, stack_to_provider_models_map: Dict[str, str]):
        self.stack_to_provider_models_map = stack_to_provider_models_map
        self.registered_models = []

    def map_to_provider_model(self, identifier: str) -> str:
        model = resolve_model(identifier)
        if not model:
            raise ValueError(f"Unknown model: `{identifier}`")

        if identifier not in self.stack_to_provider_models_map:
            raise ValueError(
                f"Model {identifier} not found in map {self.stack_to_provider_models_map}"
            )

        return self.stack_to_provider_models_map[identifier]

    async def register_model(self, model: ModelDef) -> None:
        existing = await self.get_model(model.identifier)
        if existing is not None:
            return

        if model.identifier not in self.stack_to_provider_models_map:
            raise ValueError(
                f"Unsupported model {model.identifier}. Supported models: {self.stack_to_provider_models_map.keys()}"
            )

        self.registered_models.append(model)

    async def list_models(self) -> List[ModelDef]:
        return self.registered_models

    async def get_model(self, identifier: str) -> Optional[ModelDef]:
        for model in self.registered_models:
            if model.identifier == identifier:
                return model
        return None
