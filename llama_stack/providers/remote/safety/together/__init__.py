# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import TogetherProviderDataValidator, TogetherSafetyConfig  # noqa: F401


async def get_adapter_impl(config: TogetherSafetyConfig, _deps):
    from .together import TogetherSafetyImpl

    assert isinstance(
        config, TogetherSafetyConfig
    ), f"Unexpected config type: {type(config)}"
    impl = TogetherSafetyImpl(config)
    await impl.initialize()
    return impl
