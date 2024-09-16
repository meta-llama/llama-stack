# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_toolchain.core.datatypes import *  # noqa: F403


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.telemetry,
            provider_id="console",
            pip_packages=[],
            module="llama_toolchain.telemetry.console",
            config_class="llama_toolchain.telemetry.console.ConsoleConfig",
        ),
    ]
