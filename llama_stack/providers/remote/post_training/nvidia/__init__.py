# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict

from llama_stack.distribution.datatypes import Api, ProviderSpec

from .config import NvidiaPostTrainingConfig

# post_training api and the torchtune provider is still experimental and under heavy development


async def get_provider_impl(
    config: NvidiaPostTrainingConfig,
    deps: Dict[Api, ProviderSpec],
):
    from .post_training import NvidiaPostTrainingImpl

    impl = NvidiaPostTrainingImpl(config)
    return impl
