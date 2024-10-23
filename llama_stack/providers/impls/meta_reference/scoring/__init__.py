# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import MetaReferenceScoringConfig


async def get_provider_impl(
    config: MetaReferenceScoringConfig,
    _deps,
):
    from .scoring import MetaReferenceScoringImpl

    impl = MetaReferenceScoringImpl(config)
    await impl.initialize()
    return impl
