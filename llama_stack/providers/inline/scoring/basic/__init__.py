# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Dict

from llama_stack.distribution.datatypes import Api, ProviderSpec

from .config import BasicScoringConfig


async def get_provider_impl(
    config: BasicScoringConfig,
    deps: Dict[Api, ProviderSpec],
):
    from .scoring import BasicScoringImpl

    impl = BasicScoringImpl(
        config,
        deps[Api.datasetio],
        deps[Api.datasets],
    )
    await impl.initialize()
    return impl
