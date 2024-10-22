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
            api=Api.datasetio,
            provider_type="meta-reference",
            pip_packages=["pandas"],
            module="llama_stack.providers.impls.meta_reference.datasetio",
            config_class="llama_stack.providers.impls.meta_reference.datasetio.MetaReferenceDatasetIOConfig",
            api_dependencies=[],
        ),
    ]
