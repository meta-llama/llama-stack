# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.distribution.datatypes import Api

from .config import HFilabPostTrainingConfig


async def get_provider_impl(
    config: HFilabPostTrainingConfig,
    deps: dict[Api, Any],
):
    from .post_training import HFilabPostTrainingImpl

    impl = HFilabPostTrainingImpl(
        config,
        deps[Api.datasetio],
        deps[Api.datasets],
    )
    return impl
