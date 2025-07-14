# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import logging

from llama_api_client import AsyncLlamaAPIClient

from llama_stack.providers.remote.inference.llama_openai_compat.config import LlamaCompatConfig
from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin

from .models import MODEL_ENTRIES

logger = logging.getLogger(__name__)


class LlamaCompatInferenceAdapter(LiteLLMOpenAIMixin):
    _config: LlamaCompatConfig

    def __init__(self, config: LlamaCompatConfig):
        LiteLLMOpenAIMixin.__init__(
            self,
            model_entries=MODEL_ENTRIES,
            api_key_from_config=config.api_key,
            provider_data_api_key_field="llama_api_key",
            openai_compat_api_base=config.openai_compat_api_base,
        )
        self.config = config
        self._llama_api_client = AsyncLlamaAPIClient(api_key=config.api_key)

    async def query_available_models(self) -> list[str]:
        """Query available models from the Llama API."""
        try:
            available_models = await self._llama_api_client.models.list()
            logger.info(f"Available models from Llama API: {available_models}")
            return [model.id for model in available_models]
        except Exception as e:
            logger.warning(f"Failed to query available models from Llama API: {e}")
            return []

    async def initialize(self):
        await super().initialize()

    async def shutdown(self):
        await super().shutdown()
