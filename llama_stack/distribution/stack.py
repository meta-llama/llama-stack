# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import importlib.resources
import logging
import os
import re
from typing import Any, Dict, Optional

import yaml
from termcolor import colored

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
from llama_stack.apis.safety import Safety
from llama_stack.apis.scoring import Scoring
from llama_stack.apis.scoring_functions import ScoringFunctions
from llama_stack.apis.shields import Shields
from llama_stack.apis.synthetic_data_generation import SyntheticDataGeneration
from llama_stack.apis.telemetry import Telemetry
from llama_stack.apis.tools import RAGToolRuntime, ToolGroups, ToolRuntime
from llama_stack.apis.uploads import Uploads
from llama_stack.apis.vector_dbs import VectorDBs
from llama_stack.apis.vector_io import VectorIO
from llama_stack.distribution.datatypes import StackRunConfig
from llama_stack.distribution.distribution import get_provider_registry
from llama_stack.distribution.resolver import ProviderRegistry, resolve_impls
from llama_stack.distribution.store.registry import create_dist_registry
from llama_stack.providers.datatypes import Api

log = logging.getLogger(__name__)


class LlamaStack(
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
    Uploads,
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


async def register_resources(run_config: StackRunConfig, impls: Dict[Api, Any]):
    for rsrc, api, register_method, list_method in RESOURCES:
        objects = getattr(run_config, rsrc)
        if api not in impls:
            continue

        method = getattr(impls[api], register_method)
        for obj in objects:
            await method(**obj.model_dump())

        method = getattr(impls[api], list_method)
        response = await method()

        objects_to_process = response.data if hasattr(response, "data") else response

        for obj in objects_to_process:
            log.info(
                f"{rsrc.capitalize()}: {colored(obj.identifier, 'white', attrs=['bold'])} served by {colored(obj.provider_id, 'white', attrs=['bold'])}",
            )

    log.info("")


class EnvVarError(Exception):
    def __init__(self, var_name: str, path: str = ""):
        self.var_name = var_name
        self.path = path
        super().__init__(f"Environment variable '{var_name}' not set or empty{f' at {path}' if path else ''}")


def redact_sensitive_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Redact sensitive information from config before printing."""
    sensitive_patterns = ["api_key", "api_token", "password", "secret"]

    def _redact_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = _redact_dict(v)
            elif isinstance(v, list):
                result[k] = [_redact_dict(i) if isinstance(i, dict) else i for i in v]
            elif any(pattern in k.lower() for pattern in sensitive_patterns):
                result[k] = "********"
            else:
                result[k] = v
        return result

    return _redact_dict(data)


def replace_env_vars(config: Any, path: str = "") -> Any:
    if isinstance(config, dict):
        result = {}
        for k, v in config.items():
            try:
                result[k] = replace_env_vars(v, f"{path}.{k}" if path else k)
            except EnvVarError as e:
                raise EnvVarError(e.var_name, e.path) from None
        return result

    elif isinstance(config, list):
        result = []
        for i, v in enumerate(config):
            try:
                result.append(replace_env_vars(v, f"{path}[{i}]"))
            except EnvVarError as e:
                raise EnvVarError(e.var_name, e.path) from None
        return result

    elif isinstance(config, str):
        pattern = r"\${env\.([A-Z0-9_]+)(?::([^}]*))?}"

        def get_env_var(match):
            env_var = match.group(1)
            default_val = match.group(2)

            value = os.environ.get(env_var)
            if not value:
                if default_val is None:
                    raise EnvVarError(env_var, path)
                else:
                    value = default_val

            # expand "~" from the values
            return os.path.expanduser(value)

        try:
            return re.sub(pattern, get_env_var, config)
        except EnvVarError as e:
            raise EnvVarError(e.var_name, e.path) from None

    return config


def validate_env_pair(env_pair: str) -> tuple[str, str]:
    """Validate and split an environment variable key-value pair."""
    try:
        key, value = env_pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key in environment variable pair: {env_pair}")
        if not all(c.isalnum() or c == "_" for c in key):
            raise ValueError(f"Key must contain only alphanumeric characters and underscores: {key}")
        return key, value
    except ValueError as e:
        raise ValueError(
            f"Invalid environment variable format '{env_pair}': {str(e)}. Expected format: KEY=value"
        ) from e


# Produces a stack of providers for the given run config. Not all APIs may be
# asked for in the run config.
async def construct_stack(
    run_config: StackRunConfig, provider_registry: Optional[ProviderRegistry] = None
) -> Dict[Api, Any]:
    dist_registry, _ = await create_dist_registry(run_config.metadata_store, run_config.image_name)
    impls = await resolve_impls(run_config, provider_registry or get_provider_registry(), dist_registry)
    await register_resources(run_config, impls)
    return impls


def get_stack_run_config_from_template(template: str) -> StackRunConfig:
    template_path = importlib.resources.files("llama_stack") / f"templates/{template}/run.yaml"

    with importlib.resources.as_file(template_path) as path:
        if not path.exists():
            raise ValueError(f"Template '{template}' not found at {template_path}")
        run_config = yaml.safe_load(path.open())

    return StackRunConfig(**replace_env_vars(run_config))
