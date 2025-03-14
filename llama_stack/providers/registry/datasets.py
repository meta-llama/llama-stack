# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.providers.datatypes import (
    AdapterSpec,
    Api,
    ProviderSpec,
    remote_provider_spec,
)


def available_providers() -> List[ProviderSpec]:
    return [
        remote_provider_spec(
            api=Api.datasets,
            adapter=AdapterSpec(
                adapter_type="nvidia",
                pip_packages=[
                    "datasets",
                ],
                module="llama_stack.providers.remote.datasets.nvidia",
                config_class="llama_stack.providers.remote.datasets.nvidia.NvidiaDatasetConfig",
            ),
        ),
    ]
