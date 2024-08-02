# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List, Optional

from llama_toolchain.inference.adapters import available_inference_adapters

from .datatypes import ApiSurface, Distribution

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
    "hydra-core",
    "hydra-zen",
    "json-strong-typing",
    "llama-models",
    "omegaconf",
    "pandas",
    "Pillow",
    "pydantic==1.10.13",
    "pydantic_core==2.18.2",
    "python-openapi",
    "requests",
    "tiktoken",
    "torch",
    "transformers",
    "uvicorn",
]


def available_distributions() -> List[Distribution]:
    inference_adapters_by_id = {a.adapter_id: a for a in available_inference_adapters()}

    return [
        Distribution(
            name="local-source",
            description="Use code from `llama_toolchain` itself to serve all llama stack APIs",
            additional_pip_packages=COMMON_DEPENDENCIES,
            adapters={
                ApiSurface.inference: inference_adapters_by_id["meta-reference"],
            },
        ),
        Distribution(
            name="local-ollama",
            description="Like local-source, but use ollama for running LLM inference",
            additional_pip_packages=COMMON_DEPENDENCIES,
            adapters={
                ApiSurface.inference: inference_adapters_by_id["meta-ollama"],
            },
        ),
    ]


def resolve_distribution(name: str) -> Optional[Distribution]:
    for dist in available_distributions():
        if dist.name == name:
            return dist
    return None
