# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import copy
from typing import Any, Dict, List

from pydantic import BaseModel

from llama_stack.apis.providers import ListProvidersResponse, ProviderInfo, Providers
from llama_stack.log import get_logger

from .datatypes import Provider, StackRunConfig
from .stack import redact_sensitive_fields

logger = get_logger(name=__name__, category="core")


class ProviderImplConfig(BaseModel):
    run_config: StackRunConfig


async def get_provider_impl(config, deps):
    impl = ProviderImpl(config, deps)
    await impl.initialize()
    return impl


class ProviderImpl(Providers):
    def __init__(self, config, deps):
        self.config = config
        self.deps = deps

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        logger.debug("ProviderImpl.shutdown")
        pass

    async def list_providers(self) -> ListProvidersResponse:
        run_config = self.config.run_config
        safe_config = StackRunConfig(**redact_sensitive_fields(run_config.model_dump()))
        ret = []
        for api, providers in safe_config.providers.items():
            ret.extend(
                [
                    ProviderInfo(
                        api=api,
                        provider_id=p.provider_id,
                        provider_type=p.provider_type,
                        config=p.config,
                    )
                    for p in providers
                ]
            )

        return ListProvidersResponse(data=ret)

    async def inspect_provider(self, provider_id: str) -> ProviderInfo:
        all_providers = await self.list_providers()
        for p in all_providers.data:
            if p.provider_id == provider_id:
                return p

        raise ValueError(f"Provider {provider_id} not found")

    async def update_provider(
        self, api: str, provider_id: str, provider_type: str, config: Dict[str, Any]
    ) -> ProviderInfo:
        # config = ast.literal_eval(provider_request.config)
        prov = Provider(
            provider_id=provider_id,
            provider_type=provider_type,
            config=config,
        )
        existing_provider = None
        # if the provider isn't there or the API is invalid, we should not continue
        for prov_api, providers in self.config.run_config.providers.items():
            if prov_api != api:
                continue
            for p in providers:
                if p.provider_id == provider_id:
                    existing_provider = p
                    break
            if existing_provider is not None:
                break

        if existing_provider is None:
            raise ValueError(f"Provider {provider_id} not found, you can only update already registered providers.")

        new_config = self.merge_providers(existing_provider, prov)
        existing_provider.config = new_config

        # takes a single provider, validates its in the registry
        # if it is, merge the provider config with the existing one
        ret = ProviderInfo(
            api=api,
            provider_id=prov.provider_id,
            provider_type=prov.provider_type,
            config=new_config,
        )

        return ret

    def merge_dicts(self, base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merges `overrides` into `base`, replacing only specified keys."""

        merged = copy.deepcopy(base)  # Preserve original dict
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                # Recursively merge if both are dictionaries
                merged[key] = self.merge_dicts(merged[key], value)
            else:
                # Otherwise, directly override
                merged[key] = value

        return merged

    def merge_configs(
        self, global_config: Dict[str, List[Provider]], new_config: Dict[str, List[Provider]]
    ) -> Dict[str, List[Provider]]:
        merged_config = copy.deepcopy(global_config)  # Preserve original structure

        for key, new_providers in new_config.items():
            if key in merged_config:
                existing_providers = {p.provider_id: p for p in merged_config[key]}

                for new_provider in new_providers:
                    if new_provider.provider_id in existing_providers:
                        # Override settings of existing provider
                        existing = existing_providers[new_provider.provider_id]
                        existing.config = self.merge_dicts(existing.config, new_provider.config)
                    else:
                        # Append new provider
                        merged_config[key].append(new_provider)
            else:
                # Add new category entirely
                merged_config[key] = new_providers

        return merged_config

    def merge_providers(self, current_provider: Provider, new_provider: Provider) -> Dict[str, Any]:
        return self.merge_dicts(current_provider.config, new_provider.config)
