# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict

from llama_stack.distribution.datatypes import Api, ProviderSpec

from .config import MetaReferencePostTrainingConfig


async def get_provider_impl(
    config: MetaReferencePostTrainingConfig,
    deps: Dict[Api, ProviderSpec],
):
    from .post_training import MetaReferencePostTrainingImpl

    impl = MetaReferencePostTrainingImpl(
        config,
        deps[Api.datasetio],
    )
    # await impl.initialize()
    return impl
