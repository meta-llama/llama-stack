# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import importlib.resources
import tempfile
from typing import Any

import yaml

from llama_stack.apis.agents import Agents
from llama_stack.apis.batch_inference import BatchInference
from llama_stack.apis.benchmarks import Benchmarks
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.eval import Eval
from llama_stack.apis.files import Files
from llama_stack.apis.inference import Inference
from llama_stack.apis.inspect import Inspect
from llama_stack.apis.models import Models
from llama_stack.apis.post_training import PostTraining
from llama_stack.apis.providers import Providers
from llama_stack.apis.safety import Safety
from llama_stack.apis.scoring import Scoring
from llama_stack.apis.scoring_functions import ScoringFunctions
from llama_stack.apis.shields import Shields
from llama_stack.apis.synthetic_data_generation import SyntheticDataGeneration
from llama_stack.apis.telemetry import Telemetry
from llama_stack.apis.tools import RAGToolRuntime, ToolGroups, ToolRuntime
from llama_stack.apis.vector_dbs import VectorDBs
from llama_stack.apis.vector_io import VectorIO
from llama_stack.distribution.datatypes import Provider, StackRunConfig
from llama_stack.distribution.distribution import get_provider_registry
from llama_stack.distribution.inspect import DistributionInspectConfig, DistributionInspectImpl
from llama_stack.distribution.providers import ProviderImpl, ProviderImplConfig
from llama_stack.distribution.resolver import ProviderRegistry, resolve_impls
from llama_stack.distribution.store.registry import create_dist_registry
from llama_stack.distribution.utils.dynamic import instantiate_class_type
from llama_stack.distribution.utils.env import replace_env_vars
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import Api

logger = get_logger(name=__name__, category="core")


class LlamaStack(
    Providers,
    VectorDBs,
    Inference,
    BatchInference,
    Agents,
    Safety,
    SyntheticDataGeneration,
    Datasets,
    Telemetry,
    PostTraining,
    VectorIO,
    Eval,
    Benchmarks,
    Scoring,
    ScoringFunctions,
    DatasetIO,
    Models,
    Shields,
    Inspect,
    ToolGroups,
    ToolRuntime,
    RAGToolRuntime,
    Files,
):
    pass


RESOURCES = [
    ("models", Api.models, "register_model", "list_models"),
    ("shields", Api.shields, "register_shield", "list_shields"),
    ("vector_dbs", Api.vector_dbs, "register_vector_db", "list_vector_dbs"),
    ("datasets", Api.datasets, "register_dataset", "list_datasets"),
    (
        "scoring_fns",
        Api.scoring_functions,
        "register_scoring_function",
        "list_scoring_functions",
    ),
    ("benchmarks", Api.benchmarks, "register_benchmark", "list_benchmarks"),
    ("tool_groups", Api.tool_groups, "register_tool_group", "list_tool_groups"),
]


async def register_resources(run_config: StackRunConfig, impls: dict[Api, Any]):
    for rsrc, api, register_method, list_method in RESOURCES:
        objects = getattr(run_config, rsrc)
        if api not in impls:
            continue

        method = getattr(impls[api], register_method)
        for obj in objects:
            # In complex templates, like our starter template, we may have dynamic model ids
            # given by environment variables. This allows those environment variables to have
            # a default value of __disabled__ to skip registration of the model if not set.
            if (
                hasattr(obj, "provider_model_id")
                and obj.provider_model_id is not None
                and "__disabled__" in obj.provider_model_id
            ):
                continue
            # we want to maintain the type information in arguments to method.
            # instead of method(**obj.model_dump()), which may convert a typed attr to a dict,
            # we use model_dump() to find all the attrs and then getattr to get the still typed value.
            await method(**{k: getattr(obj, k) for k in obj.model_dump().keys()})

        method = getattr(impls[api], list_method)
        response = await method()

        objects_to_process = response.data if hasattr(response, "data") else response

        for obj in objects_to_process:
            logger.debug(
                f"{rsrc.capitalize()}: {obj.identifier} served by {obj.provider_id}",
            )


def add_internal_implementations(impls: dict[Api, Any], run_config: StackRunConfig) -> None:
    """Add internal implementations (inspect and providers) to the implementations dictionary.

    Args:
        impls: Dictionary of API implementations
        run_config: Stack run configuration
    """
    inspect_impl = DistributionInspectImpl(
        DistributionInspectConfig(run_config=run_config),
        deps=impls,
    )
    impls[Api.inspect] = inspect_impl

    providers_impl = ProviderImpl(
        ProviderImplConfig(run_config=run_config),
        deps=impls,
    )
    impls[Api.providers] = providers_impl


# Produces a stack of providers for the given run config. Not all APIs may be
# asked for in the run config.
async def construct_stack(
    run_config: StackRunConfig, provider_registry: ProviderRegistry | None = None
) -> dict[Api, Any]:
    dist_registry, _ = await create_dist_registry(run_config.metadata_store, run_config.image_name)
    policy = run_config.server.auth.access_policy if run_config.server.auth else []
    impls = await resolve_impls(
        run_config, provider_registry or get_provider_registry(run_config), dist_registry, policy
    )

    # Add internal implementations after all other providers are resolved
    add_internal_implementations(impls, run_config)

    await register_resources(run_config, impls)
    return impls


def get_stack_run_config_from_template(template: str) -> StackRunConfig:
    template_path = importlib.resources.files("llama_stack") / f"templates/{template}/run.yaml"

    with importlib.resources.as_file(template_path) as path:
        if not path.exists():
            raise ValueError(f"Template '{template}' not found at {template_path}")
        run_config = yaml.safe_load(path.open())

    return StackRunConfig(**replace_env_vars(run_config))


def run_config_from_adhoc_config_spec(
    adhoc_config_spec: str, provider_registry: ProviderRegistry | None = None
) -> StackRunConfig:
    """
    Create an adhoc distribution from a list of API providers.

    The list should be of the form "api=provider", e.g. "inference=fireworks". If you have
    multiple pairs, separate them with commas or semicolons, e.g. "inference=fireworks,safety=llama-guard,agents=meta-reference"
    """

    api_providers = adhoc_config_spec.replace(";", ",").split(",")
    provider_registry = provider_registry or get_provider_registry()

    distro_dir = tempfile.mkdtemp()
    provider_configs_by_api = {}
    for api_provider in api_providers:
        api_str, provider = api_provider.split("=")
        api = Api(api_str)

        providers_by_type = provider_registry[api]
        provider_spec = providers_by_type.get(provider)
        if not provider_spec:
            provider_spec = providers_by_type.get(f"inline::{provider}")
        if not provider_spec:
            provider_spec = providers_by_type.get(f"remote::{provider}")

        if not provider_spec:
            raise ValueError(
                f"Provider {provider} (or remote::{provider} or inline::{provider}) not found for API {api}"
            )

        # call method "sample_run_config" on the provider spec config class
        provider_config_type = instantiate_class_type(provider_spec.config_class)
        provider_config = replace_env_vars(provider_config_type.sample_run_config(__distro_dir__=distro_dir))

        provider_configs_by_api[api_str] = [
            Provider(
                provider_id=provider,
                provider_type=provider_spec.provider_type,
                config=provider_config,
            )
        ]
    config = StackRunConfig(
        image_name="distro-test",
        apis=list(provider_configs_by_api.keys()),
        providers=provider_configs_by_api,
    )
    return config
