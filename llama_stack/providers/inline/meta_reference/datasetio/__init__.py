# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import MetaReferenceDatasetIOConfig


async def get_provider_impl(
    config: MetaReferenceDatasetIOConfig,
    _deps,
):
    from .datasetio import MetaReferenceDatasetIOImpl

    impl = MetaReferenceDatasetIOImpl(config)
    await impl.initialize()
    return impl
