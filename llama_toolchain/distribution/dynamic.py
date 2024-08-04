# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import importlib
from typing import Any, Dict

from .datatypes import Adapter, PassthroughApiAdapter, SourceAdapter


def instantiate_class_type(fully_qualified_name):
    module_name, class_name = fully_qualified_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


# returns a class implementing the protocol corresponding to the ApiSurface
def instantiate_adapter(
    adapter: SourceAdapter, adapter_config: Dict[str, Any], deps: Dict[str, Adapter]
):
    module = importlib.import_module(adapter.module)

    config_type = instantiate_class_type(adapter.config_class)
    config = config_type(**adapter_config)
    return asyncio.run(module.get_adapter_impl(config, deps))


def instantiate_client(adapter: PassthroughApiAdapter, base_url: str):
    module = importlib.import_module(adapter.module)

    return asyncio.run(module.get_client_impl(base_url))
