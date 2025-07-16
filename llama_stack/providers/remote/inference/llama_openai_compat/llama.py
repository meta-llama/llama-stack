# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import logging

from llama_api_client import AsyncLlamaAPIClient, NotFoundError

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

    async def check_model_availability(self, model: str) -> bool:
        """
        Check if a specific model is available from Llama API.

        :param model: The model identifier to check.
        :return: True if the model is available dynamically, False otherwise.
        """
        try:
            llama_api_client = self._get_llama_api_client()
            retrieved_model = await llama_api_client.models.retrieve(model)
            logger.info(f"Model {retrieved_model.id} is available from Llama API")
            return True

        except NotFoundError:
            logger.error(f"Model {model} is not available from Llama API")
            return False

        except Exception as e:
            logger.error(f"Failed to check model availability from Llama API: {e}")
            return False

    async def initialize(self):
        await super().initialize()

    async def shutdown(self):
        await super().shutdown()

    def _get_llama_api_client(self) -> AsyncLlamaAPIClient:
        return AsyncLlamaAPIClient(api_key=self.get_api_key(), base_url=self.config.openai_compat_api_base)
