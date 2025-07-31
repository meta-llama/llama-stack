# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import requests

from llama_stack.apis.models import Model
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin

from .config import SambaNovaImplConfig
from .models import MODEL_ENTRIES

logger = get_logger(name=__name__, category="inference")


class SambaNovaInferenceAdapter(LiteLLMOpenAIMixin):
    _config: SambaNovaImplConfig

    def __init__(self, config: SambaNovaImplConfig):
        self.config = config
        self.environment_available_models = []
        LiteLLMOpenAIMixin.__init__(
            self,
            model_entries=MODEL_ENTRIES,
            litellm_provider_name="sambanova",
            api_key_from_config=self.config.api_key.get_secret_value() if self.config.api_key else None,
            provider_data_api_key_field="sambanova_api_key",
            openai_compat_api_base=self.config.url,
            download_images=True,  # SambaNova requires base64 image encoding
            json_schema_strict=False,  # SambaNova doesn't support strict=True yet
        )

    async def register_model(self, model: Model) -> Model:
        model_id = self.get_provider_model_id(model.provider_resource_id)

        list_models_url = self.config.url + "/models"
        if len(self.environment_available_models) == 0:
            try:
                response = requests.get(list_models_url)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Request to {list_models_url} failed") from e
            self.environment_available_models = [model.get("id") for model in response.json().get("data", {})]

        if model_id.split("sambanova/")[-1] not in self.environment_available_models:
            logger.warning(f"Model {model_id} not available in {list_models_url}")
        return model

    async def initialize(self):
        await super().initialize()

    async def shutdown(self):
        await super().shutdown()
