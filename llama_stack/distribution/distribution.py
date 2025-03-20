# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import importlib
import os
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel

from llama_stack.distribution.datatypes import StackRunConfig
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import (
    AdapterSpec,
    Api,
    ProviderSpec,
    remote_provider_spec,
)

logger = get_logger(name=__name__, category="core")


def stack_apis() -> List[Api]:
    return list(Api)


class AutoRoutedApiInfo(BaseModel):
    routing_table_api: Api
    router_api: Api


def builtin_automatically_routed_apis() -> List[AutoRoutedApiInfo]:
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


def providable_apis() -> List[Api]:
    routing_table_apis = {x.routing_table_api for x in builtin_automatically_routed_apis()}
    return [api for api in Api if api not in routing_table_apis and api != Api.inspect and api != Api.providers]


def get_provider_registry(config: Optional[StackRunConfig] = None) -> Dict[Api, Dict[str, ProviderSpec]]:
    ret = {}
    for api in providable_apis():
        name = api.name.lower()
        logger.info(f"Importing module {name}")
        module = importlib.import_module(f"llama_stack.providers.registry.{name}")
        ret[api] = {a.provider_type: a for a in module.available_providers()}

    # Load external provider specs from YAML files if specified
    # Walk through the external providers directory and load the provider specs
    # for each API. A provider spec is a YAML file that contains the provider type,
    # the adapter, and the API dependencies. Here is a tree of the directory structure:
    #
    # external_providers/
    #   inference/
    #     ollama.yaml
    #     vllm.yaml
    #   shields/
    #     llama-guard.yaml
    #     openai.yaml
    #   vector_io/
    #     qdrant.yaml
    #     pinecone.yaml
    #     weaviate.yaml
    #   models/
    if config and config.external_providers_dir:
        external_providers_dir = os.path.abspath(config.external_providers_dir)
        logger.info(f"Loading external providers from {external_providers_dir}")

        for api in providable_apis():
            api_dir = os.path.join(external_providers_dir, api.name.lower())
            if not os.path.exists(api_dir):
                logger.debug(f"No external provider directory found for {api.name}")
                continue

            # Look for provider spec files in the API directory
            for filename in os.listdir(api_dir):
                if not filename.endswith(".yaml"):
                    logger.debug(f"Skipping non-YAML file {filename}")
                    continue

                provider_type = os.path.splitext(filename)[0]
                spec_path = os.path.abspath(os.path.join(api_dir, filename))
                logger.info(f"Loading provider(s) spec from {spec_path}")

                try:
                    with open(spec_path, "r") as f:
                        spec_data = yaml.safe_load(f)

                    # Convert YAML data to ProviderSpec
                    adapter = AdapterSpec(**spec_data["adapter"])
                    spec = remote_provider_spec(
                        api=api,
                        adapter=adapter,
                        api_dependencies=[Api(dep) for dep in spec_data.get("api_dependencies", [])],
                    )
                    provider_type = f"remote::{adapter.adapter_type}"
                    logger.info(f"Loaded remote provider spec for {provider_type} from {spec_path}")
                    ret[api][provider_type] = spec
                except Exception as e:
                    logger.error(f"Failed to load provider spec from {spec_path}: {e}")
    return ret
