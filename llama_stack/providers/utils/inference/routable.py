# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict, List

from llama_models.sku_list import resolve_model

from llama_stack.distribution.datatypes import RoutableProvider


class RoutableProviderForModels(RoutableProvider):

    def __init__(self, stack_to_provider_models_map: Dict[str, str]):
        self.stack_to_provider_models_map = stack_to_provider_models_map

    async def validate_routing_keys(self, routing_keys: List[str]):
        for routing_key in routing_keys:
            if routing_key not in self.stack_to_provider_models_map:
                raise ValueError(
                    f"Routing key {routing_key} not found in map {self.stack_to_provider_models_map}"
                )

    def map_to_provider_model(self, routing_key: str) -> str:
        model = resolve_model(routing_key)
        if not model:
            raise ValueError(f"Unknown model: `{routing_key}`")

        if routing_key not in self.stack_to_provider_models_map:
            raise ValueError(
                f"Model {routing_key} not found in map {self.stack_to_provider_models_map}"
            )

        return self.stack_to_provider_models_map[routing_key]
