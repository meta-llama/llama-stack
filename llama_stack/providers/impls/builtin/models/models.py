# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import asyncio

from typing import AsyncIterator, Union

from llama_models.llama3.api.datatypes import StopReason
from llama_models.sku_list import resolve_model

from llama_stack.apis.models import *  # noqa: F403
from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_models.datatypes import CoreModelId, Model
from llama_models.sku_list import resolve_model

from llama_stack.distribution.datatypes import (
    Api,
    GenericProviderConfig,
    StackRunConfig,
)
from termcolor import cprint


class BuiltinModelsImpl(Models):
    def __init__(
        self,
        config: StackRunConfig,
    ) -> None:
        print("BuiltinModelsImpl init")

        self.run_config = config
        self.models = {}

        print("BuiltinModelsImpl run_config", config)

        # check against inference & safety api
        apis_with_models = [Api.inference, Api.safety]

        for api in apis_with_models:
            # check against provider_map (simple case single model)
            if api.value in config.provider_map:
                provider_spec = config.provider_map[api.value]
                core_model_id = provider_spec.config
                print("provider_spec", provider_spec)
                model_spec = ModelServingSpec(
                    provider_config=provider_spec,
                )
                # get supported model ids  from the provider
                supported_model_ids = self.get_supported_model_ids(provider_spec)
                for model_id in supported_model_ids:
                    self.models[model_id] = ModelServingSpec(
                        llama_model=resolve_model(model_id),
                        provider_config=provider_spec,
                        api=api.value,
                    )

            # check against provider_routing_table (router with multiple models)
            # with routing table, we use the routing_key as the supported models

    def resolve_supported_model_ids(self) -> list[CoreModelId]:
        # TODO: for remote providers, provide registry to list supported models

        return ["Meta-Llama3.1-8B-Instruct"]

    async def initialize(self) -> None:
        pass

    async def list_models(self) -> ModelsListResponse:
        pass
        # return ModelsListResponse(models_list=list(self.models.values()))

    async def get_model(self, core_model_id: str) -> ModelsGetResponse:
        pass
        # if core_model_id in self.models:
        #     return ModelsGetResponse(core_model_spec=self.models[core_model_id])
        # raise RuntimeError(f"Cannot find {core_model_id} in model registry")
