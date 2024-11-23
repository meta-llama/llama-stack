# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import SsambanovaImplConfig
from .ssambanova import SsambanovaInferenceAdapter


async def get_adapter_impl(config: SsambanovaImplConfig, _deps):
    assert isinstance(
        config, SsambanovaImplConfig
    ), f"Unexpected config type: {type(config)}"
    impl = SsambanovaInferenceAdapter(config)
    await impl.initialize()
    return impl
