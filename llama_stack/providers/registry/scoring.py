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
            api=Api.scoring,
            provider_type="inline::basic",
            pip_packages=[],
            module="llama_stack.providers.inline.scoring.basic",
            config_class="llama_stack.providers.inline.scoring.basic.BasicScoringConfig",
            api_dependencies=[
                Api.datasetio,
                Api.datasets,
            ],
        ),
        InlineProviderSpec(
            api=Api.scoring,
            provider_type="inline::llm-as-judge",
            pip_packages=[],
            module="llama_stack.providers.inline.scoring.llm_as_judge",
            config_class="llama_stack.providers.inline.scoring.llm_as_judge.LlmAsJudgeScoringConfig",
            api_dependencies=[
                Api.datasetio,
                Api.datasets,
                Api.inference,
            ],
        ),
        InlineProviderSpec(
            api=Api.scoring,
            provider_type="inline::braintrust",
            pip_packages=["autoevals", "openai"],
            module="llama_stack.providers.inline.scoring.braintrust",
            config_class="llama_stack.providers.inline.scoring.braintrust.BraintrustScoringConfig",
            api_dependencies=[
                Api.datasetio,
                Api.datasets,
            ],
            provider_data_validator="llama_stack.providers.inline.scoring.braintrust.BraintrustProviderDataValidator",
        ),
    ]
