# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import LocalfsDatasetIOConfig


async def get_provider_impl(
    config: LocalfsDatasetIOConfig,
    _deps,
):
    from .datasetio import LocalfsDatasetIOImpl

    impl = LocalfsDatasetIOImpl(config)
    await impl.initialize()
    return impl
