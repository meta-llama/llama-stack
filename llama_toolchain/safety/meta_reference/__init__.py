# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import SafetyConfig


async def get_provider_impl(config: SafetyConfig, _deps):
    from .safety import MetaReferenceSafetyImpl

    assert isinstance(config, SafetyConfig), f"Unexpected config type: {type(config)}"

    impl = MetaReferenceSafetyImpl(config)
    await impl.initialize()
    return impl
