# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from functools import lru_cache
from typing import List, Optional

from .datatypes import *  # noqa: F403
from .distribution import api_providers


@lru_cache()
def available_distribution_specs() -> List[DistributionSpec]:
    providers = api_providers()
    return [
        DistributionSpec(
            spec_id="local",
            description="Use code from `llama_toolchain` itself to serve all llama stack APIs",
            provider_specs={
                Api.inference: providers[Api.inference]["meta-reference"],
                Api.memory: providers[Api.memory]["meta-reference-faiss"],
                Api.safety: providers[Api.safety]["meta-reference"],
                Api.agentic_system: providers[Api.agentic_system]["meta-reference"],
            },
        ),
        DistributionSpec(
            spec_id="remote",
            description="Point to remote services for all llama stack APIs",
            provider_specs={x: remote_provider_spec(x) for x in providers},
        ),
        DistributionSpec(
            spec_id="local-ollama",
            description="Like local, but use ollama for running LLM inference",
            provider_specs={
                # this is ODD; make this easier -- we just need a better function to retrieve registered providers
                Api.inference: providers[Api.inference][remote_provider_id("ollama")],
                Api.safety: providers[Api.safety]["meta-reference"],
                Api.agentic_system: providers[Api.agentic_system]["meta-reference"],
                Api.memory: providers[Api.memory]["meta-reference-faiss"],
            },
        ),
        DistributionSpec(
            spec_id="test-agentic",
            description="Test agentic with others as remote",
            provider_specs={
                Api.agentic_system: providers[Api.agentic_system]["meta-reference"],
                Api.inference: remote_provider_spec(Api.inference),
                Api.memory: remote_provider_spec(Api.memory),
                Api.safety: remote_provider_spec(Api.safety),
            },
        ),
        DistributionSpec(
            spec_id="test-inference",
            description="Test inference provider",
            provider_specs={
                Api.inference: providers[Api.inference]["meta-reference"],
            },
        ),
        DistributionSpec(
            spec_id="test-memory",
            description="Test memory provider",
            provider_specs={
                Api.inference: providers[Api.inference]["meta-reference"],
                Api.memory: providers[Api.memory]["meta-reference-faiss"],
            },
        ),
        DistributionSpec(
            spec_id="test-safety",
            description="Test safety provider",
            provider_specs={
                Api.safety: providers[Api.safety]["meta-reference"],
            },
        ),
    ]


@lru_cache()
def resolve_distribution_spec(spec_id: str) -> Optional[DistributionSpec]:
    for spec in available_distribution_specs():
        if spec.spec_id == spec_id:
            return spec
    return None
