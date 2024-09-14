# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from functools import lru_cache
from typing import List, Optional

from .datatypes import *  # noqa: F403


@lru_cache()
def available_distribution_specs() -> List[DistributionSpec]:
    return [
        DistributionSpec(
            distribution_type="local",
            description="Use code from `llama_toolchain` itself to serve all llama stack APIs",
            providers={
                "inference": "meta-reference",
                "memory": "meta-reference-faiss",
                "safety": "meta-reference",
                "agentic_system": "meta-reference",
                "telemetry": "console",
            },
        ),
        DistributionSpec(
            distribution_type="local-ollama",
            description="Like local, but use ollama for running LLM inference",
            providers={
                "inference": remote_provider_type("ollama"),
                "safety": "meta-reference",
                "agentic_system": "meta-reference",
                "memory": "meta-reference-faiss",
                "telemetry": "console",
            },
        ),
        DistributionSpec(
            distribution_type="local-plus-fireworks-inference",
            description="Use Fireworks.ai for running LLM inference",
            providers={
                "inference": remote_provider_type("fireworks"),
                "safety": "meta-reference",
                "agentic_system": "meta-reference",
                "memory": "meta-reference-faiss",
                "telemetry": "console",
            },
        ),
        DistributionSpec(
            distribution_type="local-plus-tgi-inference",
            description="Use TGI for running LLM inference",
            providers={
                "inference": remote_provider_type("tgi"),
                "safety": "meta-reference",
                "agentic_system": "meta-reference",
                "memory": "meta-reference-faiss",
            },
        ),
    ]


@lru_cache()
def resolve_distribution_spec(
    distribution_type: str,
) -> Optional[DistributionSpec]:
    for spec in available_distribution_specs():
        if spec.distribution_type == distribution_type:
            return spec
    return None
