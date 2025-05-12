# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.providers.datatypes import ProviderContext

from .config import VLLMConfig


async def get_provider_impl(context: ProviderContext, config: VLLMConfig, deps: dict[str, Any]):
    from .vllm import VLLMInferenceImpl

    impl = VLLMInferenceImpl(context, config)
    await impl.initialize()
    return impl
