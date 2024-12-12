# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Dict

from pydantic import BaseModel

from llama_stack.distribution.datatypes import Api, ProviderSpec

from .config import BraintrustScoringConfig


class BraintrustProviderDataValidator(BaseModel):
    openai_api_key: str


async def get_provider_impl(
    config: BraintrustScoringConfig,
    deps: Dict[Api, ProviderSpec],
):
    from .braintrust import BraintrustScoringImpl

    impl = BraintrustScoringImpl(config, deps[Api.datasetio], deps[Api.datasets])
    await impl.initialize()
    return impl
