# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from pydantic import BaseModel

from llama_stack.apis.providers import ListProvidersResponse, ProviderInfo, Providers

from .datatypes import StackRunConfig
from .stack import redact_sensitive_fields


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

    async def list_providers(self) -> ListProvidersResponse:
        run_config = self.config.run_config
        ret = []
        for api, providers in run_config.providers.items():
            ret.extend(
                [
                    ProviderInfo(
                        api=api,
                        provider_id=p.provider_id,
                        provider_type=p.provider_type,
                    )
                    for p in providers
                ]
            )

        return ListProvidersResponse(data=ret)

    async def inspect_provider(self, provider_id: str) -> Optional[ProviderInfo]:
        run_config = self.config.run_config
        safe_config = StackRunConfig(**redact_sensitive_fields(run_config.model_dump()))
        for api, providers in safe_config.providers.items():
            for p in providers:
                if p.provider_id == provider_id:
                    return ProviderInfo(
                        api=api,
                        provider_id=p.provider_id,
                        provider_type=p.provider_type,
                    )

        return None
