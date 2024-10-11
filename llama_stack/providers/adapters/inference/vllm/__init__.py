# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import VLLMImplConfig
from .vllm import VLLMInferenceAdapter


async def get_adapter_impl(config: VLLMImplConfig, _deps):
    assert isinstance(config, VLLMImplConfig), f"Unexpected config type: {type(config)}"
    impl = VLLMInferenceAdapter(config)
    await impl.initialize()
    return impl
