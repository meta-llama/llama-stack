# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from unittest.mock import AsyncMock, MagicMock, patch

from llama_stack.distribution.stack import replace_env_vars
from llama_stack.providers.remote.inference.openai.config import OpenAIConfig
from llama_stack.providers.remote.inference.openai.openai import OpenAIInferenceAdapter


class TestOpenAIBaseURLConfig:
    """Test that OPENAI_BASE_URL environment variable properly configures the OpenAI adapter."""

    def test_default_base_url_without_env_var(self):
        """Test that the adapter uses the default OpenAI base URL when no environment variable is set."""
        config = OpenAIConfig(api_key="test-key")
        adapter = OpenAIInferenceAdapter(config)

        assert adapter.get_base_url() == "https://api.openai.com/v1"

    def test_custom_base_url_from_config(self):
        """Test that the adapter uses a custom base URL when provided in config."""
        custom_url = "https://custom.openai.com/v1"
        config = OpenAIConfig(api_key="test-key", base_url=custom_url)
        adapter = OpenAIInferenceAdapter(config)

        assert adapter.get_base_url() == custom_url

    @patch.dict(os.environ, {"OPENAI_BASE_URL": "https://env.openai.com/v1"})
    def test_base_url_from_environment_variable(self):
        """Test that the adapter uses base URL from OPENAI_BASE_URL environment variable."""
        # Use sample_run_config which has proper environment variable syntax
        config_data = OpenAIConfig.sample_run_config(api_key="test-key")
        processed_config = replace_env_vars(config_data)
        config = OpenAIConfig.model_validate(processed_config)
        adapter = OpenAIInferenceAdapter(config)

        assert adapter.get_base_url() == "https://env.openai.com/v1"

    @patch.dict(os.environ, {"OPENAI_BASE_URL": "https://env.openai.com/v1"})
    def test_config_overrides_environment_variable(self):
        """Test that explicit config value overrides environment variable."""
        custom_url = "https://config.openai.com/v1"
        config = OpenAIConfig(api_key="test-key", base_url=custom_url)
        adapter = OpenAIInferenceAdapter(config)

        # Config should take precedence over environment variable
        assert adapter.get_base_url() == custom_url

    @patch("llama_stack.providers.utils.inference.openai_mixin.AsyncOpenAI")
    def test_client_uses_configured_base_url(self, mock_openai_class):
        """Test that the OpenAI client is initialized with the configured base URL."""
        custom_url = "https://test.openai.com/v1"
        config = OpenAIConfig(api_key="test-key", base_url=custom_url)
        adapter = OpenAIInferenceAdapter(config)

        # Mock the get_api_key method since it's delegated to LiteLLMOpenAIMixin
        adapter.get_api_key = MagicMock(return_value="test-key")

        # Access the client property to trigger AsyncOpenAI initialization
        _ = adapter.client

        # Verify AsyncOpenAI was called with the correct base_url
        mock_openai_class.assert_called_once_with(
            api_key="test-key",
            base_url=custom_url,
        )

    @patch("llama_stack.providers.utils.inference.openai_mixin.AsyncOpenAI")
    async def test_check_model_availability_uses_configured_url(self, mock_openai_class):
        """Test that check_model_availability uses the configured base URL."""
        custom_url = "https://test.openai.com/v1"
        config = OpenAIConfig(api_key="test-key", base_url=custom_url)
        adapter = OpenAIInferenceAdapter(config)

        # Mock the get_api_key method
        adapter.get_api_key = MagicMock(return_value="test-key")

        # Mock the AsyncOpenAI client and its models.retrieve method
        mock_client = MagicMock()
        mock_client.models.retrieve = AsyncMock(return_value=MagicMock())
        mock_openai_class.return_value = mock_client

        # Call check_model_availability and verify it returns True
        assert await adapter.check_model_availability("gpt-4")

        # Verify the client was created with the custom URL
        mock_openai_class.assert_called_with(
            api_key="test-key",
            base_url=custom_url,
        )

        # Verify the method was called and returned True
        mock_client.models.retrieve.assert_called_once_with("gpt-4")

    @patch.dict(os.environ, {"OPENAI_BASE_URL": "https://proxy.openai.com/v1"})
    @patch("llama_stack.providers.utils.inference.openai_mixin.AsyncOpenAI")
    async def test_environment_variable_affects_model_availability_check(self, mock_openai_class):
        """Test that setting OPENAI_BASE_URL environment variable affects where model availability is checked."""
        # Use sample_run_config which has proper environment variable syntax
        config_data = OpenAIConfig.sample_run_config(api_key="test-key")
        processed_config = replace_env_vars(config_data)
        config = OpenAIConfig.model_validate(processed_config)
        adapter = OpenAIInferenceAdapter(config)

        # Mock the get_api_key method
        adapter.get_api_key = MagicMock(return_value="test-key")

        # Mock the AsyncOpenAI client
        mock_client = MagicMock()
        mock_client.models.retrieve = AsyncMock(return_value=MagicMock())
        mock_openai_class.return_value = mock_client

        # Call check_model_availability and verify it returns True
        assert await adapter.check_model_availability("gpt-4")

        # Verify the client was created with the environment variable URL
        mock_openai_class.assert_called_with(
            api_key="test-key",
            base_url="https://proxy.openai.com/v1",
        )
