# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from .config import TelemetryConfig, TelemetrySink
from .telemetry import TelemetryAdapter

__all__ = ["TelemetryConfig", "TelemetryAdapter", "TelemetrySink"]


async def get_provider_impl(config: TelemetryConfig, deps: Dict[str, Any]):
    impl = TelemetryAdapter(config, deps)
    await impl.initialize()
    return impl
