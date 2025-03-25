# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import RamalamaImplConfig


async def get_adapter_impl(config: RamalamaImplConfig, _deps):
    from .ramalama import RamalamaInferenceAdapter

    impl = RamalamaInferenceAdapter(config.url)
    await impl.initialize()
    return impl
