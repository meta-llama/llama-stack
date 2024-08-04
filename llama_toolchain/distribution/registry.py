# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from functools import lru_cache
from typing import List, Optional

from llama_toolchain.inference.adapters import available_inference_adapters
from llama_toolchain.safety.adapters import available_safety_adapters

from .datatypes import ApiSurface, Distribution, PassthroughApiAdapter

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


@lru_cache()
def available_distributions() -> List[Distribution]:
    inference_adapters_by_id = {a.adapter_id: a for a in available_inference_adapters()}
    safety_adapters_by_id = {a.adapter_id: a for a in available_safety_adapters()}

    return [
        Distribution(
            name="local-source",
            description="Use code from `llama_toolchain` itself to serve all llama stack APIs",
            additional_pip_packages=COMMON_DEPENDENCIES,
            adapters={
                ApiSurface.inference: inference_adapters_by_id["meta-reference"],
                ApiSurface.safety: safety_adapters_by_id["meta-reference"],
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
                ApiSurface.inference: PassthroughApiAdapter(
                    api_surface=ApiSurface.inference,
                    adapter_id="inference-passthrough",
                    base_url="http://localhost:5001",
                ),
                ApiSurface.safety: PassthroughApiAdapter(
                    api_surface=ApiSurface.safety,
                    adapter_id="safety-passthrough",
                    base_url="http://localhost:5001",
                ),
            },
        ),
        Distribution(
            name="local-ollama",
            description="Like local-source, but use ollama for running LLM inference",
            additional_pip_packages=COMMON_DEPENDENCIES,
            adapters={
                ApiSurface.inference: inference_adapters_by_id["meta-ollama"],
                ApiSurface.safety: safety_adapters_by_id["meta-reference"],
            },
        ),
    ]


@lru_cache()
def resolve_distribution(name: str) -> Optional[Distribution]:
    for dist in available_distributions():
        if dist.name == name:
            return dist
    return None
