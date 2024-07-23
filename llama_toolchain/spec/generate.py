# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime

import yaml

from pyopenapi import Info, Options, Server, Specification

from llama_models.llama3_1.api.datatypes import *  # noqa: F403
from llama_toolchain.dataset.api import *  # noqa: F403
from llama_toolchain.evaluations.api import *  # noqa: F403
from llama_toolchain.inference.api import *  # noqa: F403
from llama_toolchain.memory.api import *  # noqa: F403
from llama_toolchain.post_training.api import *  # noqa: F403
from llama_toolchain.reward_scoring.api import *  # noqa: F403
from llama_toolchain.synthetic_data_generation.api import *  # noqa: F403
from agentic_system.api import *  # noqa: F403


class LlamaStackEndpoints(
    Inference,
    AgenticSystem,
    RewardScoring,
    SyntheticDataGeneration,
    Datasets,
    PostTraining,
    MemoryBanks,
    Evaluations,
): ...


if __name__ == "__main__":
    now = str(datetime.now())
    print(
        "Converting the spec to YAML (openapi.yaml) and HTML (openapi.html) at " + now
    )
    spec = Specification(
        LlamaStackEndpoints,
        Options(
            server=Server(url="http://any-hosted-llama-stack.com"),
            info=Info(
                title="[DRAFT] Llama Stack Specification",
                version="0.0.1",
                description="""This is the specification of the llama stack that provides
                a set of endpoints and their corresponding interfaces that are tailored to
                best leverage Llama Models. The specification is still in draft and subject to change.
                Generated at """
                + now,
            ),
        ),
    )
    with open("openapi.yaml", "w", encoding="utf-8") as fp:
        yaml.dump(spec.get_json(), fp, allow_unicode=True)

    with open("openapi.html", "w") as fp:
        spec.write_html(fp, pretty_print=True)
