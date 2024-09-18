# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import RedisImplConfig


async def get_adapter_impl(config: RedisImplConfig, _deps):
    from .redis import RedisControlPlaneAdapter

    impl = RedisControlPlaneAdapter(config)
    await impl.initialize()
    return impl
