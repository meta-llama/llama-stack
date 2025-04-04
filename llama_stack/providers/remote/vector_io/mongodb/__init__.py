# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict

from llama_stack.providers.datatypes import Api, ProviderSpec

from .config import MongoDBVectorIOConfig


async def get_adapter_impl(config: MongoDBVectorIOConfig, deps: Dict[Api, ProviderSpec]):
    from .mongodb import MongoDBVectorIOAdapter      

    impl = MongoDBVectorIOAdapter(config, deps[Api.inference])
    await impl.initialize()
    return impl
