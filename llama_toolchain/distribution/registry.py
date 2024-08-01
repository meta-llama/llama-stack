# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from .datatypes import LlamaStackDistribution


def all_registered_distributions() -> List[LlamaStackDistribution]:
    return [
        LlamaStackDistribution(
            name="local-source",
            description="Use code within `llama_toolchain` itself to run model inference and everything on top",
            pip_packages=[],
        ),
        LlamaStackDistribution(
            name="local-ollama",
            description="Like local-source, but use ollama for running LLM inference",
            pip_packages=[],
        ),
    ]
