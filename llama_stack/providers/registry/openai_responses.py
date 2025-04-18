# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.providers.datatypes import Api, InlineProviderSpec, ProviderSpec


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.openai_responses,
            provider_type="inline::openai-responses",
            pip_packages=[],
            module="llama_stack.providers.inline.openai_responses",
            config_class="llama_stack.providers.inline.openai_responses.config.OpenAIResponsesImplConfig",
            api_dependencies=[
                Api.models,
                Api.inference,
                Api.tool_groups,
                Api.tool_runtime,
            ],
        ),
    ]
