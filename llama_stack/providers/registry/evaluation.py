# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.providers.datatypes import Api, InlineProviderSpec, ProviderSpec
from llama_stack.providers.utils.kvstore import kvstore_dependencies


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.evaluation,
            provider_type="inline::meta-reference",
            pip_packages=[
                "matplotlib",
                "pillow",
                "pandas",
                "scikit-learn",
            ]
            + kvstore_dependencies(),
            module="llama_stack.providers.inline.evaluation.meta_reference",
            config_class="llama_stack.providers.inline.evaluation.meta_reference.MetaReferenceEvaluationConfig",
            api_dependencies=[
                Api.inference,
                Api.agents,
                Api.datasets,
                Api.datasetio,
            ],
        ),
    ]
