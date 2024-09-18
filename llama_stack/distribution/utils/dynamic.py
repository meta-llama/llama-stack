# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import importlib
from typing import Any, Dict

from llama_stack.distribution.datatypes import *  # noqa: F403


def instantiate_class_type(fully_qualified_name):
    module_name, class_name = fully_qualified_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


# returns a class implementing the protocol corresponding to the Api
async def instantiate_provider(
    provider_spec: ProviderSpec,
    deps: Dict[str, Any],
    provider_config: ProviderMapEntry,
):
    module = importlib.import_module(provider_spec.module)

    args = []
    if isinstance(provider_spec, RemoteProviderSpec):
        if provider_spec.adapter:
            method = "get_adapter_impl"
        else:
            method = "get_client_impl"

        assert isinstance(provider_config, GenericProviderConfig)
        config_type = instantiate_class_type(provider_spec.config_class)
        config = config_type(**provider_config.config)
        args = [config, deps]
    elif isinstance(provider_spec, RouterProviderSpec):
        method = "get_router_impl"

        assert isinstance(provider_config, list)
        inner_specs = {x.provider_id: x for x in provider_spec.inner_specs}
        inner_impls = []
        for routing_entry in provider_config:
            impl = await instantiate_provider(
                inner_specs[routing_entry.provider_id],
                deps,
                routing_entry,
            )
            inner_impls.append((routing_entry.routing_key, impl))

        config = None
        args = [inner_impls, deps]
    else:
        method = "get_provider_impl"

        assert isinstance(provider_config, GenericProviderConfig)
        config_type = instantiate_class_type(provider_spec.config_class)
        config = config_type(**provider_config.config)
        args = [config, deps]

    fn = getattr(module, method)
    impl = await fn(*args)
    impl.__provider_spec__ = provider_spec
    impl.__provider_config__ = config
    return impl
