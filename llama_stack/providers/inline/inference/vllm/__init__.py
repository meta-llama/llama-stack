# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from .config import VLLMConfig


async def get_provider_impl(config: VLLMConfig, _deps) -> Any:
    from .vllm import VLLMInferenceImpl

    impl = VLLMInferenceImpl(config)
    await impl.initialize()
    return impl
