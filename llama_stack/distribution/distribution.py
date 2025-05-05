# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import glob
import importlib
import os
from typing import Any

import yaml
from pydantic import BaseModel

from llama_stack.log import get_logger
from llama_stack.providers.datatypes import (
    AdapterSpec,
    Api,
    InlineProviderSpec,
    ProviderSpec,
    remote_provider_spec,
)

logger = get_logger(name=__name__, category="core")


def stack_apis() -> list[Api]:
    return list(Api)


class AutoRoutedApiInfo(BaseModel):
    routing_table_api: Api
    router_api: Api


def builtin_automatically_routed_apis() -> list[AutoRoutedApiInfo]:
    return [
        AutoRoutedApiInfo(
            routing_table_api=Api.models,
            router_api=Api.inference,
        ),
        AutoRoutedApiInfo(
            routing_table_api=Api.shields,
            router_api=Api.safety,
        ),
        AutoRoutedApiInfo(
            routing_table_api=Api.vector_dbs,
            router_api=Api.vector_io,
        ),
        AutoRoutedApiInfo(
            routing_table_api=Api.datasets,
            router_api=Api.datasetio,
        ),
        AutoRoutedApiInfo(
            routing_table_api=Api.scoring_functions,
            router_api=Api.scoring,
        ),
        AutoRoutedApiInfo(
            routing_table_api=Api.benchmarks,
            router_api=Api.eval,
        ),
        AutoRoutedApiInfo(
            routing_table_api=Api.tool_groups,
            router_api=Api.tool_runtime,
        ),
    ]


def providable_apis() -> list[Api]:
    routing_table_apis = {x.routing_table_api for x in builtin_automatically_routed_apis()}
    return [api for api in Api if api not in routing_table_apis and api != Api.inspect and api != Api.providers]


def _load_remote_provider_spec(spec_data: dict[str, Any], api: Api) -> ProviderSpec:
    adapter = AdapterSpec(**spec_data["adapter"])
    spec = remote_provider_spec(
        api=api,
        adapter=adapter,
        api_dependencies=[Api(dep) for dep in spec_data.get("api_dependencies", [])],
    )
    return spec


def _load_inline_provider_spec(spec_data: dict[str, Any], api: Api, provider_name: str) -> ProviderSpec:
    spec = InlineProviderSpec(
        api=api,
        provider_type=f"inline::{provider_name}",
        pip_packages=spec_data.get("pip_packages", []),
        module=spec_data["module"],
        config_class=spec_data["config_class"],
        api_dependencies=[Api(dep) for dep in spec_data.get("api_dependencies", [])],
        optional_api_dependencies=[Api(dep) for dep in spec_data.get("optional_api_dependencies", [])],
        provider_data_validator=spec_data.get("provider_data_validator"),
        container_image=spec_data.get("container_image"),
    )
    return spec


def get_provider_registry(
    config=None,
) -> dict[Api, dict[str, ProviderSpec]]:
    """Get the provider registry, optionally including external providers.

    This function loads both built-in providers and external providers from YAML files.
    External providers are loaded from a directory structure like:

    providers.d/
      remote/
        inference/
          custom_ollama.yaml
          vllm.yaml
        vector_io/
          qdrant.yaml
        safety/
          llama-guard.yaml
      inline/
        inference/
          custom_ollama.yaml
          vllm.yaml
        vector_io/
          qdrant.yaml
        safety/
          llama-guard.yaml

    Args:
        config: Optional object containing the external providers directory path

    Returns:
        A dictionary mapping APIs to their available providers

    Raises:
        FileNotFoundError: If the external providers directory doesn't exist
        ValueError: If any provider spec is invalid
    """

    ret: dict[Api, dict[str, ProviderSpec]] = {}
    for api in providable_apis():
        name = api.name.lower()
        logger.debug(f"Importing module {name}")
        try:
            module = importlib.import_module(f"llama_stack.providers.registry.{name}")
            ret[api] = {a.provider_type: a for a in module.available_providers()}
        except ImportError as e:
            logger.warning(f"Failed to import module {name}: {e}")

    # Check if config has the external_providers_dir attribute
    if config and hasattr(config, "external_providers_dir") and config.external_providers_dir:
        external_providers_dir = os.path.abspath(config.external_providers_dir)
        if not os.path.exists(external_providers_dir):
            raise FileNotFoundError(f"External providers directory not found: {external_providers_dir}")
        logger.info(f"Loading external providers from {external_providers_dir}")

        for api in providable_apis():
            api_name = api.name.lower()

            # Process both remote and inline providers
            for provider_type in ["remote", "inline"]:
                api_dir = os.path.join(external_providers_dir, provider_type, api_name)
                if not os.path.exists(api_dir):
                    logger.debug(f"No {provider_type} provider directory found for {api_name}")
                    continue

                # Look for provider spec files in the API directory
                for spec_path in glob.glob(os.path.join(api_dir, "*.yaml")):
                    provider_name = os.path.splitext(os.path.basename(spec_path))[0]
                    logger.info(f"Loading {provider_type} provider spec from {spec_path}")

                    try:
                        with open(spec_path) as f:
                            spec_data = yaml.safe_load(f)

                        if provider_type == "remote":
                            spec = _load_remote_provider_spec(spec_data, api)
                            provider_type_key = f"remote::{provider_name}"
                        else:
                            spec = _load_inline_provider_spec(spec_data, api, provider_name)
                            provider_type_key = f"inline::{provider_name}"

                        logger.info(f"Loaded {provider_type} provider spec for {provider_type_key} from {spec_path}")
                        if provider_type_key in ret[api]:
                            logger.warning(f"Overriding already registered provider {provider_type_key} for {api.name}")
                        ret[api][provider_type_key] = spec
                    except yaml.YAMLError as yaml_err:
                        logger.error(f"Failed to parse YAML file {spec_path}: {yaml_err}")
                        raise yaml_err
                    except Exception as e:
                        logger.error(f"Failed to load provider spec from {spec_path}: {e}")
                        raise e
    return ret
