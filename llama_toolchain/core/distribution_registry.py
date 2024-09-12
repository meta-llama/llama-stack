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
                Api.inference: "meta-reference",
                Api.memory: "meta-reference-faiss",
                Api.safety: "meta-reference",
                Api.agentic_system: "meta-reference",
            },
        ),
        DistributionSpec(
            distribution_type="remote",
            description="Point to remote services for all llama stack APIs",
            providers={x: "remote" for x in Api},
        ),
        DistributionSpec(
            distribution_type="local-ollama",
            description="Like local, but use ollama for running LLM inference",
            providers={
                Api.inference: remote_provider_type("ollama"),
                Api.safety: "meta-reference",
                Api.agentic_system: "meta-reference",
                Api.memory: "meta-reference-faiss",
            },
        ),
        DistributionSpec(
            distribution_type="local-plus-fireworks-inference",
            description="Use Fireworks.ai for running LLM inference",
            providers={
                Api.inference: remote_provider_type("fireworks"),
                Api.safety: "meta-reference",
                Api.agentic_system: "meta-reference",
                Api.memory: "meta-reference-faiss",
            },
        ),
        DistributionSpec(
            distribution_type="local-plus-together-inference",
            description="Use Together.ai for running LLM inference",
            providers={
                Api.inference: remote_provider_type("together"),
                Api.safety: "meta-reference",
                Api.agentic_system: "meta-reference",
                Api.memory: "meta-reference-faiss",
            },
        ),
        DistributionSpec(
            distribution_type="local-plus-tgi-inference",
            description="Use TGI for running LLM inference",
            providers={
                Api.inference: remote_provider_type("tgi"),
                Api.safety: "meta-reference",
                Api.agentic_system: "meta-reference",
                Api.memory: "meta-reference-faiss",
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
