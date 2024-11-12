# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from termcolor import colored

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.agents import *  # noqa: F403
from llama_stack.apis.datasets import *  # noqa: F403
from llama_stack.apis.datasetio import *  # noqa: F403
from llama_stack.apis.scoring import *  # noqa: F403
from llama_stack.apis.scoring_functions import *  # noqa: F403
from llama_stack.apis.eval import *  # noqa: F403
from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.apis.batch_inference import *  # noqa: F403
from llama_stack.apis.memory import *  # noqa: F403
from llama_stack.apis.telemetry import *  # noqa: F403
from llama_stack.apis.post_training import *  # noqa: F403
from llama_stack.apis.synthetic_data_generation import *  # noqa: F403
from llama_stack.apis.safety import *  # noqa: F403
from llama_stack.apis.models import *  # noqa: F403
from llama_stack.apis.memory_banks import *  # noqa: F403
from llama_stack.apis.shields import *  # noqa: F403
from llama_stack.apis.inspect import *  # noqa: F403
from llama_stack.apis.eval_tasks import *  # noqa: F403

from llama_stack.distribution.datatypes import StackRunConfig
from llama_stack.distribution.distribution import get_provider_registry
from llama_stack.distribution.resolver import resolve_impls
from llama_stack.distribution.store.registry import create_dist_registry
from llama_stack.providers.datatypes import Api


class LlamaStack(
    MemoryBanks,
    Inference,
    BatchInference,
    Agents,
    Safety,
    SyntheticDataGeneration,
    Datasets,
    Telemetry,
    PostTraining,
    Memory,
    Eval,
    EvalTasks,
    Scoring,
    ScoringFunctions,
    DatasetIO,
    Models,
    Shields,
    Inspect,
):
    pass


# Produces a stack of providers for the given run config. Not all APIs may be
# asked for in the run config.
async def construct_stack(run_config: StackRunConfig) -> Dict[Api, Any]:
    dist_registry, _ = await create_dist_registry(
        run_config.metadata_store, run_config.image_name
    )

    impls = await resolve_impls(run_config, get_provider_registry(), dist_registry)

    resources = [
        ("models", Api.models, "register_model", "list_models"),
        ("shields", Api.shields, "register_shield", "list_shields"),
        ("memory_banks", Api.memory_banks, "register_memory_bank", "list_memory_banks"),
        ("datasets", Api.datasets, "register_dataset", "list_datasets"),
        (
            "scoring_fns",
            Api.scoring_functions,
            "register_scoring_function",
            "list_scoring_functions",
        ),
        ("eval_tasks", Api.eval_tasks, "register_eval_task", "list_eval_tasks"),
    ]
    for rsrc, api, register_method, list_method in resources:
        objects = getattr(run_config, rsrc)
        if api not in impls:
            continue

        method = getattr(impls[api], register_method)
        for obj in objects:
            await method(**obj.model_dump())

        method = getattr(impls[api], list_method)
        for obj in await method():
            print(
                f"{rsrc.capitalize()}: {colored(obj.identifier, 'white', attrs=['bold'])} served by {colored(obj.provider_id, 'white', attrs=['bold'])}",
            )

    print("")
    return impls
