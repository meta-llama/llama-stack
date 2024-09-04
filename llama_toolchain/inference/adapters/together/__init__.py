# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import TogetherImplConfig


async def get_adapter_impl(config: TogetherImplConfig, _deps):
    from .together import TogetherInferenceAdapter

    assert isinstance(
        config, TogetherImplConfig
    ), f"Unexpected config type: {type(config)}"
    impl = TogetherInferenceAdapter(config)
    await impl.initialize()
    return impl
