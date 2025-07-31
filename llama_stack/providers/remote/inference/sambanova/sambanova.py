# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin

from .config import SambaNovaImplConfig
from .models import MODEL_ENTRIES


class SambaNovaInferenceAdapter(LiteLLMOpenAIMixin):
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
