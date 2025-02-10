# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from .config import SampleVectorIOConfig


async def get_adapter_impl(config: SampleVectorIOConfig, _deps) -> Any:
    from .sample import SampleVectorIOImpl

    impl = SampleVectorIOImpl(config)
    await impl.initialize()
    return impl
