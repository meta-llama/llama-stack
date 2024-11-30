# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.utils.telemetry.jaeger import JaegerTraceStore
from .config import OpenTelemetryConfig


async def get_adapter_impl(config: OpenTelemetryConfig, deps):
    from .opentelemetry import OpenTelemetryAdapter

    trace_store = JaegerTraceStore(config.jaeger_query_endpoint, config.service_name)
    impl = OpenTelemetryAdapter(config, trace_store, deps)
    await impl.initialize()
    return impl
