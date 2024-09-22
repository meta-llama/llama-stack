# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import asyncio

from typing import AsyncIterator, Union

from llama_models.llama3.api.datatypes import StopReason
from llama_models.sku_list import resolve_model

from llama_stack.distribution.distribution import Api, api_providers

from llama_stack.apis.models import *  # noqa: F403
from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_models.datatypes import CoreModelId, Model
from llama_models.sku_list import resolve_model

from llama_stack.distribution.datatypes import *  # noqa: F403
from termcolor import cprint


class BuiltinModelsImpl(Models):
    def __init__(
        self,
        config: StackRunConfig,
    ) -> None:
        self.run_config = config
        self.models = {}
        # check against inference & safety api
        apis_with_models = [Api.inference, Api.safety]

        all_providers = api_providers()

        for api in apis_with_models:

            # check against provider_map (simple case single model)
            if api.value in config.provider_map:
                providers_for_api = all_providers[api]
                provider_spec = config.provider_map[api.value]
                core_model_id = provider_spec.config
                # get supported model ids  from the provider
                supported_model_ids = self.get_supported_model_ids(
                    api.value, provider_spec, providers_for_api
                )
                for model_id in supported_model_ids:
                    self.models[model_id] = ModelServingSpec(
                        llama_model=resolve_model(model_id),
                        provider_config=provider_spec,
                        api=api.value,
                    )

            # check against provider_routing_table (router with multiple models)
            # with routing table, we use the routing_key as the supported models
            if api.value in config.provider_routing_table:
                routing_table = config.provider_routing_table[api.value]
                for rt_entry in routing_table:
                    model_id = rt_entry.routing_key
                    self.models[model_id] = ModelServingSpec(
                        llama_model=resolve_model(model_id),
                        provider_config=GenericProviderConfig(
                            provider_id=rt_entry.provider_id,
                            config=rt_entry.config,
                        ),
                        api=api.value,
                    )

        print("BuiltinModelsImpl models", self.models)

    def get_supported_model_ids(
        self,
        api: str,
        provider_spec: GenericProviderConfig,
        providers_for_api: Dict[str, ProviderSpec],
    ) -> List[str]:
        serving_models_list = []
        if api == Api.inference.value:
            provider_id = provider_spec.provider_id
            if provider_id == "meta-reference":
                serving_models_list.append(provider_spec.config["model"])
            if provider_id in {
                remote_provider_id("ollama"),
                remote_provider_id("fireworks"),
                remote_provider_id("together"),
            }:
                adapter_supported_models = providers_for_api[
                    provider_id
                ].adapter.supported_model_ids
                serving_models_list.extend(adapter_supported_models)
        elif api == Api.safety.value:
            if provider_spec.config and "llama_guard_shield" in provider_spec.config:
                llama_guard_shield = provider_spec.config["llama_guard_shield"]
                serving_models_list.append(llama_guard_shield["model"])
            if provider_spec.config and "prompt_guard_shield" in provider_spec.config:
                prompt_guard_shield = provider_spec.config["prompt_guard_shield"]
                serving_models_list.append(prompt_guard_shield["model"])
        else:
            raise NotImplementedError(f"Unsupported api {api} for builtin models")

        return serving_models_list

    async def initialize(self) -> None:
        pass

    async def list_models(self) -> ModelsListResponse:
        return ModelsListResponse(models_list=list(self.models.values()))

    async def get_model(self, core_model_id: str) -> ModelsGetResponse:
        if core_model_id in self.models:
            return ModelsGetResponse(core_model_spec=self.models[core_model_id])
        print(f"Cannot find {core_model_id} in model registry")
        return ModelsGetResponse()
