# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import SqliteControlPlaneConfig


async def get_provider_impl(config: SqliteControlPlaneConfig, _deps):
    from .control_plane import SqliteControlPlane

    impl = SqliteControlPlane(config)
    await impl.initialize()
    return impl
