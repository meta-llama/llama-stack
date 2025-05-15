# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.distribution.datatypes import Api
from llama_stack.providers.datatypes import ProviderContext

from .config import TelemetryConfig, TelemetrySink

__all__ = ["TelemetryConfig", "TelemetrySink"]


async def get_provider_impl(context: ProviderContext, config: TelemetryConfig, deps: dict[Api, Any]):
    from .telemetry import TelemetryAdapter

    impl = TelemetryAdapter(context, config, deps)
    await impl.initialize()
    return impl
