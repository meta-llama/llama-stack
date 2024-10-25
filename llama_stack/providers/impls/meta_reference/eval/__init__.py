# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Dict

from llama_stack.distribution.datatypes import Api, ProviderSpec

from .config import MetaReferenceEvalConfig


async def get_provider_impl(
    config: MetaReferenceEvalConfig,
    deps: Dict[Api, ProviderSpec],
):
    from .eval import MetaReferenceEvalImpl

    impl = MetaReferenceEvalImpl(
        config,
        deps[Api.datasetio],
        deps[Api.datasets],
        deps[Api.scoring],
        deps[Api.inference],
    )
    await impl.initialize()
    return impl
