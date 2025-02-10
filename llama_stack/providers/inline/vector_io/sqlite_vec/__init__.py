# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict
from llama_stack.providers.datatypes import Api, ProviderSpec
from .config import SQLiteVecImplConfig


async def get_provider_impl(config: SQLiteVecImplConfig, deps: Dict[Api, ProviderSpec]):
    from .sqlite_vec import SQLiteVecVectorIOImpl

    assert isinstance(config, SQLiteVecImplConfig), f"Unexpected config type: {type(config)}"
    impl = SQLiteVecVectorIOImpl(config, deps[Api.inference])
    await impl.initialize()
    return impl
