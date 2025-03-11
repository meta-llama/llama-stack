# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from llama_stack.providers.datatypes import Api

from .config import SQLiteVectorIOConfig


async def get_provider_impl(config: SQLiteVectorIOConfig, deps: Dict[Api, Any]):
    from .sqlite_vec import SQLiteVecVectorIOAdapter

    assert isinstance(config, SQLiteVectorIOConfig), f"Unexpected config type: {type(config)}"
    impl = SQLiteVecVectorIOAdapter(config, deps[Api.inference])
    await impl.initialize()
    return impl
