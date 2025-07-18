# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.models.models import Model
from llama_stack.providers.remote.inference.llamacpp.config import LlamaCppImplConfig
from llama_stack.providers.utils.inference.litellm_openai_mixin import (
    LiteLLMOpenAIMixin,
)


class LlamaCppInferenceAdapter(LiteLLMOpenAIMixin):
    _config: LlamaCppImplConfig

    def __init__(self, config: LlamaCppImplConfig):
        LiteLLMOpenAIMixin.__init__(
            self,
            model_entries=[],  # llama.cpp can work with any GGUF model
            api_key_from_config=config.api_key,
            provider_data_api_key_field="llamacpp_api_key",
            openai_compat_api_base=config.openai_compat_api_base,
        )
        self.config = config

    async def register_model(self, model: Model) -> Model:
        # llama.cpp can work with any GGUF model, so we accept any model name
        # without validation against a predefined list
        return model

    async def initialize(self):
        await super().initialize()

    async def shutdown(self):
        await super().shutdown()
