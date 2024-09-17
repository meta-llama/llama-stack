# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.distribution.datatypes import *  # noqa: F403


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.control_plane,
            provider_id="sqlite",
            pip_packages=["aiosqlite"],
            module="llama_stack.providers.impls.sqlite.control_plane",
            config_class="llama_stack.providers.impls.sqlite.control_plane.SqliteControlPlaneConfig",
        ),
        remote_provider_spec(
            Api.control_plane,
            AdapterSpec(
                adapter_id="redis",
                pip_packages=["redis"],
                module="llama_stack.providers.adapters.control_plane.redis",
            ),
        ),
    ]
