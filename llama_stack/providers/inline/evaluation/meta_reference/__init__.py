# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any, Dict

from llama_stack.distribution.datatypes import Api

from .config import MetaReferenceEvaluationConfig


async def get_provider_impl(
    config: MetaReferenceEvaluationConfig,
    deps: Dict[Api, Any],
):
    from .evaluation import MetaReferenceEvaluationImpl

    impl = MetaReferenceEvaluationImpl(
        config,
        deps[Api.datasetio],
        deps[Api.datasets],
        deps[Api.inference],
        deps[Api.agents],
    )
    await impl.initialize()
    return impl
