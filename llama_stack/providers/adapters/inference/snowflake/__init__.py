# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import SnowflakeImplConfig
from .snowflake import SnowflakeInferenceAdapter


async def get_adapter_impl(config: SnowflakeImplConfig, _deps):
    assert isinstance(
        config, SnowflakeImplConfig
    ), f"Unexpected config type: {type(config)}"
    impl = SnowflakeInferenceAdapter(config)
    await impl.initialize()
    return impl
