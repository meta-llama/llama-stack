# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from functools import lru_cache
from typing import List, Optional

from .datatypes import Api, DistributionSpec, RemoteProviderSpec
from .distribution import api_providers

# This is currently duplicated from `requirements.txt` with a few minor changes
# dev-dependencies like "ufmt" etc. are nuked. A few specialized dependencies
# are moved to the appropriate distribution.
COMMON_DEPENDENCIES = [
    "accelerate",
    "black==24.4.2",
    "blobfile",
    "codeshield",
    "fairscale",
    "fastapi",
    "fire",
    "flake8",
    "httpx",
    "huggingface-hub",
    "json-strong-typing",
    "llama-models",
    "omegaconf",
    "pandas",
    "Pillow",
    "pydantic==1.10.13",
    "pydantic_core==2.18.2",
    "python-dotenv",
    "python-openapi",
    "requests",
    "tiktoken",
    "torch",
    "transformers",
    "uvicorn",
]


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
            spec_id="inline",
            description="Use code from `llama_toolchain` itself to serve all llama stack APIs",
            additional_pip_packages=COMMON_DEPENDENCIES,
            provider_specs={
                Api.inference: providers[Api.inference]["meta-reference"],
                Api.safety: providers[Api.safety]["meta-reference"],
                Api.agentic_system: providers[Api.agentic_system]["meta-reference"],
            },
        ),
        DistributionSpec(
            spec_id="remote",
            description="Point to remote services for all llama stack APIs",
            additional_pip_packages=[
                "python-dotenv",
                "blobfile",
                "codeshield",
                "fairscale",
                "fastapi",
                "fire",
                "httpx",
                "huggingface-hub",
                "json-strong-typing",
                "pydantic==1.10.13",
                "pydantic_core==2.18.2",
                "uvicorn",
            ],
            provider_specs={x: remote_spec(x) for x in providers},
        ),
        DistributionSpec(
            spec_id="ollama-inline",
            description="Like local-source, but use ollama for running LLM inference",
            additional_pip_packages=COMMON_DEPENDENCIES,
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
