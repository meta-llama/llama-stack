# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict

from llama_stack.distribution.datatypes import Api, ProviderSpec

from .config import BuiltinImplConfig  # noqa


async def get_provider_impl(config: BuiltinImplConfig, deps: Dict[Api, ProviderSpec]):
    from .models import BuiltinModelsImpl

    assert isinstance(
        config, BuiltinImplConfig
    ), f"Unexpected config type: {type(config)}"

    print(config)

    impl = BuiltinModelsImpl(config)
    await impl.initialize()
    return impl
