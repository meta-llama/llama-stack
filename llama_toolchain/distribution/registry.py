# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from functools import lru_cache
from typing import List, Optional

from .datatypes import Api, DistributionSpec, RemoteProviderSpec
from .distribution import api_providers


def client_module(api: Api) -> str:
    return f"llama_toolchain.{api.value}.client"


def remote_spec(api: Api) -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=api,
        provider_id=f"{api.value}-remote",
        module=client_module(api),
    )


@lru_cache()
def available_distribution_specs() -> List[DistributionSpec]:
    providers = api_providers()
    return [
        DistributionSpec(
            spec_id="local",
            description="Use code from `llama_toolchain` itself to serve all llama stack APIs",
            provider_specs={
                Api.inference: providers[Api.inference]["meta-reference"],
                Api.safety: providers[Api.safety]["meta-reference"],
                Api.agentic_system: providers[Api.agentic_system]["meta-reference"],
            },
        ),
        DistributionSpec(
            spec_id="remote",
            description="Point to remote services for all llama stack APIs",
            provider_specs={x: remote_spec(x) for x in providers},
        ),
        DistributionSpec(
            spec_id="local-ollama",
            description="Like local, but use ollama for running LLM inference",
            provider_specs={
                Api.inference: providers[Api.inference]["meta-ollama"],
                Api.safety: providers[Api.safety]["meta-reference"],
                Api.agentic_system: providers[Api.agentic_system]["meta-reference"],
            },
        ),
    ]


@lru_cache()
def resolve_distribution_spec(spec_id: str) -> Optional[DistributionSpec]:
    for spec in available_distribution_specs():
        if spec.spec_id == spec_id:
            return spec
    return None
