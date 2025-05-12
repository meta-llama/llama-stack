# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.providers.datatypes import Api, ProviderContext

from .config import SQLiteVectorIOConfig


async def get_provider_impl(context: ProviderContext, config: SQLiteVectorIOConfig, deps: dict[Api, Any]):
    from .sqlite_vec import SQLiteVecVectorIOAdapter

    assert isinstance(config, SQLiteVectorIOConfig), f"Unexpected config type: {type(config)}"
    impl = SQLiteVecVectorIOAdapter(context, config, deps[Api.inference])
    await impl.initialize()
    return impl
