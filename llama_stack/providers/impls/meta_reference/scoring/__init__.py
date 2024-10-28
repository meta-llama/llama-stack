# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Dict

from llama_stack.distribution.datatypes import Api, ProviderSpec

from .config import MetaReferenceScoringConfig


async def get_provider_impl(
    config: MetaReferenceScoringConfig,
    deps: Dict[Api, ProviderSpec],
):
    from .scoring import MetaReferenceScoringImpl

    impl = MetaReferenceScoringImpl(
        config, deps[Api.datasetio], deps[Api.datasets], deps[Api.inference]
    )
    await impl.initialize()
    return impl
