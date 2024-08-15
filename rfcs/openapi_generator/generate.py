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

import inspect

from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, List, Tuple

import fire
import yaml
from llama_models import schema_utils
from pyopenapi import Info, operations, Options, Server, Specification

# We do a series of monkey-patching to ensure our definitions only use the minimal
# (json_schema_type, webmethod) definitions from the llama_models package. For
# generation though, we need the full definitions and implementations from the
#  (python-openapi, json-strong-typing) packages.

from strong_typing.schema import json_schema_type
from termcolor import colored


# PATCH `json_schema_type` first
schema_utils.json_schema_type = json_schema_type

from llama_models.llama3_1.api.datatypes import *  # noqa: F403
from llama_toolchain.agentic_system.api import *  # noqa: F403
from llama_toolchain.dataset.api import *  # noqa: F403
from llama_toolchain.evaluations.api import *  # noqa: F403
from llama_toolchain.inference.api import *  # noqa: F403
from llama_toolchain.memory.api import *  # noqa: F403
from llama_toolchain.post_training.api import *  # noqa: F403
from llama_toolchain.reward_scoring.api import *  # noqa: F403
from llama_toolchain.synthetic_data_generation.api import *  # noqa: F403


def patched_get_endpoint_functions(
    endpoint: type, prefixes: List[str]
) -> Iterator[Tuple[str, str, str, Callable]]:
    if not inspect.isclass(endpoint):
        raise ValueError(f"object is not a class type: {endpoint}")

    functions = inspect.getmembers(endpoint, inspect.isfunction)
    for func_name, func_ref in functions:
        webmethod = getattr(func_ref, "__webmethod__", None)
        if not webmethod:
            continue

        print(f"Processing {colored(func_name, 'white')}...")
        operation_name = func_name
        if operation_name.startswith("get_") or operation_name.endswith("/get"):
            prefix = "get"
        elif (
            operation_name.startswith("delete_")
            or operation_name.startswith("remove_")
            or operation_name.endswith("/delete")
            or operation_name.endswith("/remove")
        ):
            prefix = "delete"
        else:
            if webmethod.method == "GET":
                prefix = "get"
            elif webmethod.method == "DELETE":
                prefix = "delete"
            else:
                # by default everything else is a POST
                prefix = "post"

        yield prefix, operation_name, func_name, func_ref


operations._get_endpoint_functions = patched_get_endpoint_functions


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
    with open(output_dir / "llama-stack-spec.yaml", "w", encoding="utf-8") as fp:
        yaml.dump(spec.get_json(), fp, allow_unicode=True)

    with open(output_dir / "llama-stack-spec.html", "w") as fp:
        spec.write_html(fp, pretty_print=True)


if __name__ == "__main__":
    fire.Fire(main)
