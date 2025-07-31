# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import requests

from llama_stack.apis.inference import (
    ChatCompletionRequest,
    JsonSchemaResponseFormat,
    ToolChoice,
)
from llama_stack.apis.models import Model
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin
from llama_stack.providers.utils.inference.openai_compat import (
    convert_message_to_openai_dict_new,
    convert_tooldef_to_openai_tool,
    get_sampling_options,
)

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
        )

    def _get_api_key(self) -> str:
        config_api_key = self.config.api_key if self.config.api_key else None
        if config_api_key:
            return config_api_key.get_secret_value()
        else:
            provider_data = self.get_request_provider_data()
            if provider_data is None or not provider_data.sambanova_api_key:
                raise ValueError(
                    'Pass Sambanova API Key in the header X-LlamaStack-Provider-Data as { "sambanova_api_key": <your api key> }'
                )
            return provider_data.sambanova_api_key

    async def _get_params(self, request: ChatCompletionRequest) -> dict:
        input_dict = {}

        input_dict["messages"] = [
            await convert_message_to_openai_dict_new(m, download_images=True) for m in request.messages
        ]
        if fmt := request.response_format:
            if not isinstance(fmt, JsonSchemaResponseFormat):
                raise ValueError(
                    f"Unsupported response format: {type(fmt)}. Only JsonSchemaResponseFormat is supported."
                )

            fmt = fmt.json_schema
            name = fmt["title"]
            del fmt["title"]
            fmt["additionalProperties"] = False

            # Apply additionalProperties: False recursively to all objects
            fmt = self._add_additional_properties_recursive(fmt)

            input_dict["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "schema": fmt,
                    "strict": False,
                },
            }
        if request.tools:
            input_dict["tools"] = [convert_tooldef_to_openai_tool(tool) for tool in request.tools]
            if request.tool_config.tool_choice:
                input_dict["tool_choice"] = (
                    request.tool_config.tool_choice.value
                    if isinstance(request.tool_config.tool_choice, ToolChoice)
                    else request.tool_config.tool_choice
                )

        provider_data = self.get_request_provider_data()
        key_field = self.provider_data_api_key_field
        if provider_data and getattr(provider_data, key_field, None):
            api_key = getattr(provider_data, key_field)
        else:
            api_key = self._get_api_key()

        return {
            "model": request.model,
            "api_key": api_key,
            "api_base": self.config.url,
            **input_dict,
            "stream": request.stream,
            **get_sampling_options(request.sampling_params),
        }

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
