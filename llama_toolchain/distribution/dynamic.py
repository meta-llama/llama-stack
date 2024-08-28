# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import importlib
from typing import Any, Dict

from .datatypes import InlineProviderSpec, ProviderSpec, RemoteProviderSpec


def instantiate_class_type(fully_qualified_name):
    module_name, class_name = fully_qualified_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


# returns a class implementing the protocol corresponding to the Api
def instantiate_provider(
    provider_spec: InlineProviderSpec,
    provider_config: Dict[str, Any],
    deps: Dict[str, ProviderSpec],
):
    module = importlib.import_module(provider_spec.module)

    config_type = instantiate_class_type(provider_spec.config_class)
    config = config_type(**provider_config)
    return asyncio.run(module.get_provider_impl(config, deps))


def instantiate_client(
    provider_spec: RemoteProviderSpec, provider_config: Dict[str, Any]
):
    module = importlib.import_module(provider_spec.module)

    adapter = provider_spec.adapter
    if adapter is not None:
        if "adapter" not in provider_config:
            raise ValueError(
                f"Adapter is specified but not present in provider config: {provider_config}"
            )
            adapter_config = provider_config["adapter"]

        config_type = instantiate_class_type(adapter.config_class)
        if not issubclass(config_type, RemoteProviderConfig):
            raise ValueError(
                f"Config class {adapter.config_class} does not inherit from RemoteProviderConfig"
            )

        config = config_type(**adapter_config)
    else:
        config = RemoteProviderConfig(**provider_config)

    return asyncio.run(module.get_adapter_impl(config))
