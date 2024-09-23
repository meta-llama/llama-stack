# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import OpenTelemetryConfig


async def get_adapter_impl(config: OpenTelemetryConfig, _deps):
    from .opentelemetry import OpenTelemetryAdapter

    impl = OpenTelemetryAdapter(config)
    await impl.initialize()
    return impl
