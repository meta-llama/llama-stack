# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import copy
from typing import Any

from pydantic import BaseModel

from llama_stack.apis.providers import ListProvidersResponse, ProviderInfo, Providers
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import HealthResponse, HealthStatus

from .datatypes import Provider, StackRunConfig
from .utils.config import redact_sensitive_fields

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
        providers_health = await self.get_providers_health()
        ret = []
        for api, providers in safe_config.providers.items():
            for p in providers:
                # Skip providers that are not enabled
                if p.provider_id is None:
                    continue
                ret.append(
                    ProviderInfo(
                        api=api,
                        provider_id=p.provider_id,
                        provider_type=p.provider_type,
                        config=p.config,
                        health=providers_health.get(api, {}).get(
                            p.provider_id,
                            HealthResponse(
                                status=HealthStatus.NOT_IMPLEMENTED, message="Provider does not implement health check"
                            ),
                        ),
                    )
                )

        return ListProvidersResponse(data=ret)

    async def inspect_provider(self, provider_id: str) -> ProviderInfo:
        all_providers = await self.list_providers()
        for p in all_providers.data:
            if p.provider_id == provider_id:
                return p

        raise ValueError(f"Provider {provider_id} not found")

    async def get_providers_health(self) -> dict[str, dict[str, HealthResponse]]:
        """Get health status for all providers.

        Returns:
            Dict[str, Dict[str, HealthResponse]]: A dictionary mapping API names to provider health statuses.
                Each API maps to a dictionary of provider IDs to their health responses.
        """
        providers_health: dict[str, dict[str, HealthResponse]] = {}
        timeout = 1.0

        async def check_provider_health(impl: Any) -> tuple[str, HealthResponse] | None:
            # Skip special implementations (inspect/providers) that don't have provider specs
            if not hasattr(impl, "__provider_spec__"):
                return None
            api_name = impl.__provider_spec__.api.name
            if not hasattr(impl, "health"):
                return (
                    api_name,
                    HealthResponse(
                        status=HealthStatus.NOT_IMPLEMENTED, message="Provider does not implement health check"
                    ),
                )

            try:
                health = await asyncio.wait_for(impl.health(), timeout=timeout)
                return api_name, health
            except TimeoutError:
                return (
                    api_name,
                    HealthResponse(
                        status=HealthStatus.ERROR, message=f"Health check timed out after {timeout} seconds"
                    ),
                )
            except Exception as e:
                return (
                    api_name,
                    HealthResponse(status=HealthStatus.ERROR, message=f"Health check failed: {str(e)}"),
                )

        # Create tasks for all providers
        tasks = [check_provider_health(impl) for impl in self.deps.values()]

        # Wait for all health checks to complete
        results = await asyncio.gather(*tasks)

        # Organize results by API and provider ID
        for result in results:
            if result is None:  # Skip special implementations
                continue
            api_name, health_response = result
            providers_health[api_name] = health_response

        return providers_health

    async def update_provider(
        self, api: str, provider_id: str, provider_type: str, config: dict[str, Any]
    ) -> ProviderInfo:
        # config = ast.literal_eval(provider_request.config)
        prov = Provider(
            provider_id=provider_id,
            provider_type=provider_type,
            config=config,
        )
        assert prov.provider_id is not None
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
        providers_health = await self.get_providers_health()
        # takes a single provider, validates its in the registry
        # if it is, merge the provider config with the existing one
        ret = ProviderInfo(
            api=api,
            provider_id=prov.provider_id,
            provider_type=prov.provider_type,
            config=new_config,
            health=providers_health.get(api, {}).get(
                p.provider_id,
                HealthResponse(status=HealthStatus.NOT_IMPLEMENTED, message="Provider does not implement health check"),
            ),
        )

        return ret

    def merge_dicts(self, base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
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
        self, global_config: dict[str, list[Provider]], new_config: dict[str, list[Provider]]
    ) -> dict[str, list[Provider]]:
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

    def merge_providers(self, current_provider: Provider, new_provider: Provider) -> dict[str, Any]:
        return self.merge_dicts(current_provider.config, new_provider.config)
