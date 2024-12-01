# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import SambaNovaImplConfig
from .sambanova import SambaNovaInferenceAdapter

async def get_adapter_impl(config: SambaNovaImplConfig, _deps):
    assert isinstance(
        config, SambaNovaImplConfig
    ), f"Unexpected config type: {type(config)}"
    impl = SambaNovaInferenceAdapter(config)
    await impl.initialize()
    return impl
