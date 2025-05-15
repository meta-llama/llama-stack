# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.distribution.datatypes import Api
from llama_stack.providers.datatypes import ProviderContext

from .config import TorchtunePostTrainingConfig

# post_training api and the torchtune provider is still experimental and under heavy development


async def get_provider_impl(
    context: ProviderContext,
    config: TorchtunePostTrainingConfig,
    deps: dict[Api, Any],
):
    from .post_training import TorchtunePostTrainingImpl

    impl = TorchtunePostTrainingImpl(
        context,
        config,
        deps[Api.datasetio],
        deps[Api.datasets],
    )
    return impl
