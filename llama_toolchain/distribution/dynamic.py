# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import importlib
from typing import Any, Dict

from .datatypes import ProviderSpec, RemoteProviderSpec


def instantiate_class_type(fully_qualified_name):
    module_name, class_name = fully_qualified_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


# returns a class implementing the protocol corresponding to the Api
def instantiate_provider(
    provider_spec: ProviderSpec,
    provider_config: Dict[str, Any],
    deps: Dict[str, ProviderSpec],
):
    module = importlib.import_module(provider_spec.module)

    config_type = instantiate_class_type(provider_spec.config_class)
    if isinstance(provider_spec, RemoteProviderSpec):
        if provider_spec.adapter:
            method = "get_adapter_impl"
        else:
            method = "get_client_impl"
    else:
        method = "get_provider_impl"

    config = config_type(**provider_config)
    fn = getattr(module, method)
    impl = asyncio.run(fn(config, deps))
    impl.__provider_spec__ = provider_spec
    impl.__provider_config__ = config
    return impl
