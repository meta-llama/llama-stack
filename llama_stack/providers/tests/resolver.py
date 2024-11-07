# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml

from llama_stack.distribution.datatypes import *  # noqa: F403
from llama_stack.distribution.configure import parse_and_maybe_upgrade_config
from llama_stack.distribution.distribution import get_provider_registry
from llama_stack.distribution.request_headers import set_request_provider_data
from llama_stack.distribution.resolver import resolve_impls
from llama_stack.distribution.store import CachedDiskDistributionRegistry
from llama_stack.providers.utils.kvstore import kvstore_impl, SqliteKVStoreConfig


async def resolve_impls_for_test_v2(
    apis: List[Api],
    providers: Dict[str, List[Provider]],
    provider_data: Optional[Dict[str, Any]] = None,
):
    run_config = dict(
        built_at=datetime.now(),
        image_name="test-fixture",
        apis=apis,
        providers=providers,
    )
    run_config = parse_and_maybe_upgrade_config(run_config)

    sqlite_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    dist_kvstore = await kvstore_impl(SqliteKVStoreConfig(db_path=sqlite_file.name))
    dist_registry = CachedDiskDistributionRegistry(dist_kvstore)
    impls = await resolve_impls(run_config, get_provider_registry(), dist_registry)

    if provider_data:
        set_request_provider_data(
            {"X-LlamaStack-ProviderData": json.dumps(provider_data)}
        )

    return impls


async def resolve_impls_for_test(api: Api, deps: List[Api] = None):
    if "PROVIDER_CONFIG" not in os.environ:
        raise ValueError(
            "You must set PROVIDER_CONFIG to a YAML file containing provider config"
        )

    with open(os.environ["PROVIDER_CONFIG"], "r") as f:
        config_dict = yaml.safe_load(f)

    providers = read_providers(api, config_dict)

    chosen = choose_providers(providers, api, deps)
    run_config = dict(
        built_at=datetime.now(),
        image_name="test-fixture",
        apis=[api] + (deps or []),
        providers=chosen,
    )
    run_config = parse_and_maybe_upgrade_config(run_config)
    impls = await resolve_impls(run_config, get_provider_registry())

    if "provider_data" in config_dict:
        provider_id = chosen[api.value][0].provider_id
        provider_data = config_dict["provider_data"].get(provider_id, {})
        if provider_data:
            set_request_provider_data(
                {"X-LlamaStack-ProviderData": json.dumps(provider_data)}
            )

    return impls


def read_providers(api: Api, config_dict: Dict[str, Any]) -> Dict[str, Any]:
    if "providers" not in config_dict:
        raise ValueError("Config file should contain a `providers` key")

    providers = config_dict["providers"]
    if isinstance(providers, dict):
        return providers
    elif isinstance(providers, list):
        return {
            api.value: providers,
        }
    else:
        raise ValueError(
            "Config file should contain a list of providers or dict(api to providers)"
        )


def choose_providers(
    providers: Dict[str, Any], api: Api, deps: List[Api] = None
) -> Dict[str, Provider]:
    chosen = {}
    if api.value not in providers:
        raise ValueError(f"No providers found for `{api}`?")
    chosen[api.value] = [pick_provider(api, providers[api.value], "PROVIDER_ID")]

    for dep in deps or []:
        if dep.value not in providers:
            raise ValueError(f"No providers specified for `{dep}` in config?")
        chosen[dep.value] = [Provider(**x) for x in providers[dep.value]]

    return chosen


def pick_provider(api: Api, providers: List[Any], key: str) -> Provider:
    providers_by_id = {x["provider_id"]: x for x in providers}
    if len(providers_by_id) == 0:
        raise ValueError(f"No providers found for `{api}` in config file")

    if key in os.environ:
        provider_id = os.environ[key]
        if provider_id not in providers_by_id:
            raise ValueError(f"Provider ID {provider_id} not found in config file")
        provider = providers_by_id[provider_id]
    else:
        provider = list(providers_by_id.values())[0]
        provider_id = provider["provider_id"]
        print(f"No provider ID specified, picking first `{provider_id}`")

    return Provider(**provider)
