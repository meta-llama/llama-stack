# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.providers.datatypes import Api

from .config import SyntheticDataKitToolRuntimeConfig


async def get_provider_impl(config: SyntheticDataKitToolRuntimeConfig, deps: dict[Api, Any]):
    from .synthetic_data_kit import SyntheticDataKitToolRuntimeImpl

    impl = SyntheticDataKitToolRuntimeImpl(config, deps[Api.files])
    await impl.initialize()
    return impl
