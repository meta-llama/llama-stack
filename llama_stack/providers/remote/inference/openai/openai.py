# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin

from .config import OpenAIConfig
from .models import MODEL_ENTRIES


class OpenAIInferenceAdapter(LiteLLMOpenAIMixin):
    def __init__(self, config: OpenAIConfig) -> None:
        LiteLLMOpenAIMixin.__init__(
            self,
            MODEL_ENTRIES,
            api_key_from_config=config.api_key,
            provider_data_api_key_field="openai_api_key",
        )
        self.config = config
        # we set is_openai_compat so users can use the canonical
        # openai model names like "gpt-4" or "gpt-3.5-turbo"
        # and the model name will be translated to litellm's
        # "openai/gpt-4" or "openai/gpt-3.5-turbo" transparently.
        # if we do not set this, users will be exposed to the
        # litellm specific model names, an abstraction leak.
        self.is_openai_compat = True

    async def initialize(self) -> None:
        await super().initialize()

    async def shutdown(self) -> None:
        await super().shutdown()
