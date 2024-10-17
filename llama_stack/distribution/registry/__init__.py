# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any

from llama_stack.providers.datatypes import Api
from .datasets.dataset import DatasetRegistryImpl


async def get_registry_impl(api: Api, _deps) -> Any:
    api_to_registry = {
        "datasets": DatasetRegistryImpl,
    }

    if api.value not in api_to_registry:
        raise ValueError(f"API {api.value} not found in registry map")

    impl = api_to_registry[api.value]()
    await impl.initialize()
    return impl
