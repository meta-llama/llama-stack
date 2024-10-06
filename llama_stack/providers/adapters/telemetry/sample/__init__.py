# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from .config import SampleConfig


async def get_adapter_impl(config: SampleConfig, _deps) -> Any:
    from .sample import SampleTelemetryImpl

    impl = SampleTelemetryImpl(config)
    await impl.initialize()
    return impl
