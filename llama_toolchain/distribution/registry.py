# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from functools import lru_cache
from typing import List, Optional

from llama_toolchain.agentic_system.adapters import available_agentic_system_adapters

from llama_toolchain.inference.adapters import available_inference_adapters
from llama_toolchain.safety.adapters import available_safety_adapters

from .datatypes import Api, Distribution, PassthroughApiAdapter

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
    "git+ssh://git@github.com/meta-llama/llama-models.git",
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


def passthrough(api: Api, port: int) -> PassthroughApiAdapter:
    return PassthroughApiAdapter(
        api=api,
        adapter_id=f"{api.value}-passthrough",
        base_url=f"http://localhost:{port}",
        module=client_module(api),
    )


@lru_cache()
def available_distributions() -> List[Distribution]:
    inference_adapters_by_id = {a.adapter_id: a for a in available_inference_adapters()}
    safety_adapters_by_id = {a.adapter_id: a for a in available_safety_adapters()}
    agentic_system_adapters_by_id = {
        a.adapter_id: a for a in available_agentic_system_adapters()
    }

    return [
        Distribution(
            name="local-source",
            description="Use code from `llama_toolchain` itself to serve all llama stack APIs",
            additional_pip_packages=COMMON_DEPENDENCIES,
            adapters={
                Api.inference: inference_adapters_by_id["meta-reference"],
                Api.safety: safety_adapters_by_id["meta-reference"],
                Api.agentic_system: agentic_system_adapters_by_id["meta-reference"],
            },
        ),
        Distribution(
            name="full-passthrough",
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
            adapters={
                Api.inference: passthrough(Api.inference, 5001),
                Api.safety: passthrough(Api.safety, 5001),
                Api.agentic_system: passthrough(Api.agentic_system, 5001),
            },
        ),
        Distribution(
            name="local-ollama",
            description="Like local-source, but use ollama for running LLM inference",
            additional_pip_packages=COMMON_DEPENDENCIES,
            adapters={
                Api.inference: inference_adapters_by_id["meta-ollama"],
                Api.safety: safety_adapters_by_id["meta-reference"],
                Api.agentic_system: agentic_system_adapters_by_id["meta-reference"],
            },
        ),
    ]


@lru_cache()
def resolve_distribution(name: str) -> Optional[Distribution]:
    for dist in available_distributions():
        if dist.name == name:
            return dist
    return None
