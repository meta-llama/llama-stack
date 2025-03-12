# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from llama_stack.providers.datatypes import Api

from .config import ChromaVectorIOConfig


async def get_provider_impl(config: ChromaVectorIOConfig, deps: Dict[Api, Any]):
    from llama_stack.providers.remote.vector_io.chroma.chroma import (
        ChromaVectorIOAdapter,
    )

    impl = ChromaVectorIOAdapter(config, deps[Api.inference])
    await impl.initialize()
    return impl
