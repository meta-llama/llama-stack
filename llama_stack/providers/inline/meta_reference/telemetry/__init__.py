# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import ConsoleConfig


async def get_provider_impl(config: ConsoleConfig, _deps):
    from .console import ConsoleTelemetryImpl

    impl = ConsoleTelemetryImpl(config)
    await impl.initialize()
    return impl
