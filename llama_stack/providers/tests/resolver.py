# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os
from datetime import datetime

import yaml

from llama_stack.distribution.datatypes import *  # noqa: F403
from llama_stack.distribution.configure import parse_and_maybe_upgrade_config
from llama_stack.distribution.request_headers import set_request_provider_data
from llama_stack.distribution.resolver import resolve_impls_with_routing


async def resolve_impls_for_test(
    api: Api,
    models: List[ModelDef] = None,
    memory_banks: List[MemoryBankDef] = None,
    shields: List[ShieldDef] = None,
):
    if "PROVIDER_CONFIG" not in os.environ:
        raise ValueError(
            "You must set PROVIDER_CONFIG to a YAML file containing provider config"
        )

    with open(os.environ["PROVIDER_CONFIG"], "r") as f:
        config_dict = yaml.safe_load(f)

    if "providers" not in config_dict:
        raise ValueError("Config file should contain a `providers` key")

    providers_by_id = {x["provider_id"]: x for x in config_dict["providers"]}
    if len(providers_by_id) == 0:
        raise ValueError("No providers found in config file")

    if "PROVIDER_ID" in os.environ:
        provider_id = os.environ["PROVIDER_ID"]
        if provider_id not in providers_by_id:
            raise ValueError(f"Provider ID {provider_id} not found in config file")
        provider = providers_by_id[provider_id]
    else:
        provider = list(providers_by_id.values())[0]
        provider_id = provider["provider_id"]
        print(f"No provider ID specified, picking first `{provider_id}`")

    models = models or []
    shields = shields or []
    memory_banks = memory_banks or []

    models = [
        ModelDef(
            **{
                **m.dict(),
                "provider_id": provider_id,
            }
        )
        for m in models
    ]
    shields = [
        ShieldDef(
            **{
                **s.dict(),
                "provider_id": provider_id,
            }
        )
        for s in shields
    ]
    memory_banks = [
        MemoryBankDef(
            **{
                **m.dict(),
                "provider_id": provider_id,
            }
        )
        for m in memory_banks
    ]
    run_config = dict(
        built_at=datetime.now(),
        image_name="test-fixture",
        apis=[api],
        providers={api.value: [Provider(**provider)]},
        models=models,
        memory_banks=memory_banks,
        shields=shields,
    )
    run_config = parse_and_maybe_upgrade_config(run_config)
    impls = await resolve_impls_with_routing(run_config)

    if "provider_data" in config_dict:
        provider_data = config_dict["provider_data"].get(provider_id, {})
        if provider_data:
            set_request_provider_data(
                {"X-LlamaStack-ProviderData": json.dumps(provider_data)}
            )

    return impls
