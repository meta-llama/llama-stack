# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging

from anthropic import AsyncAnthropic, NotFoundError

from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin

from .config import AnthropicConfig
from .models import MODEL_ENTRIES

logger = logging.getLogger(__name__)


class AnthropicInferenceAdapter(LiteLLMOpenAIMixin):
    def __init__(self, config: AnthropicConfig) -> None:
        LiteLLMOpenAIMixin.__init__(
            self,
            MODEL_ENTRIES,
            api_key_from_config=config.api_key,
            provider_data_api_key_field="anthropic_api_key",
        )
        self.config = config
        self._client: AsyncAnthropic | None = None

    async def initialize(self) -> None:
        await super().initialize()

    async def shutdown(self) -> None:
        # Clean up the client connection pool
        if self._client:
            await self._client.aclose()
            self._client = None
        await super().shutdown()

    @property
    def client(self) -> AsyncAnthropic:
        if self._client is None:
            api_key = self.config.api_key if self.config.api_key else "no-key"
            self._client = AsyncAnthropic(api_key=api_key)
        return self._client

    async def check_model_availability(self, model: str) -> bool:
        try:
            retrieved_model = await self.client.models.retrieve(model)
            logger.info(f"Model {retrieved_model.id} is available on Anthropic")
            return True

        except NotFoundError:
            logger.info(f"Model {model} was not found on Anthropic")

        except Exception as e:
            logger.error(f"Failed to check model availability for {model} on Anthropic: {e}")

        return False
