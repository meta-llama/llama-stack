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
from pathlib import Path

import fire
import yaml

from llama_models import schema_utils

from .pyopenapi.options import Options
from .pyopenapi.specification import Info, Server
from .pyopenapi.utility import Specification

# We do some monkey-patching to ensure our definitions only use the minimal
# (json_schema_type, webmethod) definitions from the llama_models package. For
# generation though, we need the full definitions and implementations from the
#  (json-strong-typing) package.

from .strong_typing.schema import json_schema_type

schema_utils.json_schema_type = json_schema_type

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.agents import *  # noqa: F403
from llama_stack.apis.dataset import *  # noqa: F403
from llama_stack.apis.evals import *  # noqa: F403
from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.apis.batch_inference import *  # noqa: F403
from llama_stack.apis.memory import *  # noqa: F403
from llama_stack.apis.telemetry import *  # noqa: F403
from llama_stack.apis.post_training import *  # noqa: F403
from llama_stack.apis.reward_scoring import *  # noqa: F403
from llama_stack.apis.synthetic_data_generation import *  # noqa: F403
from llama_stack.apis.safety import *  # noqa: F403
from llama_stack.apis.models import *  # noqa: F403
from llama_stack.apis.memory_banks import *  # noqa: F403
from llama_stack.apis.shields import *  # noqa: F403
from llama_stack.apis.inspect import *  # noqa: F403


class LlamaStack(
    MemoryBanks,
    Inference,
    BatchInference,
    Agents,
    RewardScoring,
    Safety,
    SyntheticDataGeneration,
    Datasets,
    Telemetry,
    PostTraining,
    Memory,
    Evaluations,
    Models,
    Shields,
    Inspect,
):
    pass


# TODO: this should be fixed in the generator itself so it reads appropriate annotations
STREAMING_ENDPOINTS = [
    "/agents/turn/create",
    "/inference/chat_completion",
]


def patch_sse_stream_responses(spec: Specification):
    for path, path_item in spec.document.paths.items():
        if path in STREAMING_ENDPOINTS:
            content = path_item.post.responses["200"].content.pop("application/json")
            path_item.post.responses["200"].content["text/event-stream"] = content


def main(output_dir: str):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise ValueError(f"Directory {output_dir} does not exist")

    now = str(datetime.now())
    print(
        "Converting the spec to YAML (openapi.yaml) and HTML (openapi.html) at " + now
    )
    print("")
    spec = Specification(
        LlamaStack,
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

    patch_sse_stream_responses(spec)

    with open(output_dir / "llama-stack-spec.yaml", "w", encoding="utf-8") as fp:
        yaml.dump(spec.get_json(), fp, allow_unicode=True)

    with open(output_dir / "llama-stack-spec.html", "w") as fp:
        spec.write_html(fp, pretty_print=True)


if __name__ == "__main__":
    fire.Fire(main)
