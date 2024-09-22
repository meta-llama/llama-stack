# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from llama_stack.distribution.datatypes import Api, ProviderSpec, StackRunConfig


async def get_builtin_impl(config: StackRunConfig):
    from .models import BuiltinModelsImpl

    assert isinstance(config, StackRunConfig), f"Unexpected config type: {type(config)}"

    impl = BuiltinModelsImpl(config)
    await impl.initialize()
    return impl
