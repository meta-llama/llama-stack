# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import OllamaImplConfig


async def get_adapter_impl(config: OllamaImplConfig, _deps):
    from .ollama import OllamaInferenceAdapter

    impl = OllamaInferenceAdapter(config.url, raise_on_connect_error=config.raise_on_connect_error)
    await impl.initialize()
    return impl
