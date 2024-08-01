# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from .datatypes import Distribution, DistributionConfigDefaults


def all_registered_distributions() -> List[Distribution]:
    return [
        Distribution(
            name="local-source",
            description="Use code from `llama_toolchain` itself to serve all llama stack APIs",
            pip_packages=[],
            config_defaults=DistributionConfigDefaults(
                inference={
                    "max_seq_len": 4096,
                    "max_batch_size": 1,
                },
                safety={},
            ),
        ),
        Distribution(
            name="local-ollama",
            description="Like local-source, but use ollama for running LLM inference",
            pip_packages=["ollama"],
            config_defaults=DistributionConfigDefaults(
                inference={},
                safety={},
            ),
        ),
    ]
