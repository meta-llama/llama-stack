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

from llama_stack.distribution.client import get_client_impl
from llama_stack.distribution.datatypes import StackRunConfig
from llama_stack.distribution.distribution import get_provider_registry
from llama_stack.distribution.resolver import (
    additional_protocols_map,
    api_protocol_map,
    resolve_impls,
)
from llama_stack.distribution.store.registry import create_dist_registry
from llama_stack.providers.datatypes import Api, RemoteProviderConfig


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

    impls = await maybe_get_remote_stack_impls(run_config)
    if impls is None:
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


# NOTE: this code path is really for the tests so you can send HTTP requests
# to the remote stack without needing to use llama-stack-client
async def maybe_get_remote_stack_impls(
    run_config: StackRunConfig,
) -> Optional[Dict[Api, Any]]:
    remote_config = remote_provider_config(run_config)
    if not remote_config:
        return None

    protocols = api_protocol_map()
    additional_protocols = additional_protocols_map()

    impls = {}
    for api_str in run_config.apis:
        api = Api(api_str)
        impls[api] = await get_client_impl(
            protocols[api],
            None,
            remote_config,
            {},
        )
        if api in additional_protocols:
            _, additional_protocol, additional_api = additional_protocols[api]
            impls[additional_api] = await get_client_impl(
                additional_protocol,
                None,
                remote_config,
                {},
            )

    return impls


def remote_provider_config(
    run_config: StackRunConfig,
) -> Optional[RemoteProviderConfig]:
    remote_config = None
    has_non_remote = False
    for api_providers in run_config.providers.values():
        for provider in api_providers:
            if provider.provider_type == "remote":
                remote_config = RemoteProviderConfig(**provider.config)
            else:
                has_non_remote = True

    if remote_config:
        assert not has_non_remote, "Remote stack cannot have non-remote providers"

    return remote_config
