# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from llama_stack.providers.datatypes import Api

from .config import MilvusVectorIOConfig


async def get_provider_impl(config: MilvusVectorIOConfig, deps: Dict[Api, Any]):
    from llama_stack.providers.remote.vector_io.milvus.milvus import MilvusVectorIOAdapter

    impl = MilvusVectorIOAdapter(config, deps[Api.inference])
    await impl.initialize()
    return impl
