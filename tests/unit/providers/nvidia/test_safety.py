# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_stack.apis.inference.inference import CompletionMessage, UserMessage
from llama_stack.apis.safety import RunShieldResponse, ViolationLevel
from llama_stack.apis.shields import Shield
from llama_stack.providers.remote.safety.nvidia.config import NVIDIASafetyConfig
from llama_stack.providers.remote.safety.nvidia.nvidia import NVIDIASafetyAdapter


class TestNVIDIASafetyAdapter(unittest.TestCase):
    def setUp(self):
        os.environ["NVIDIA_GUARDRAILS_URL"] = "http://nemo.test"

        # Initialize the adapter
        self.config = NVIDIASafetyConfig(
            guardrails_service_url=os.environ["NVIDIA_GUARDRAILS_URL"],
        )
        self.adapter = NVIDIASafetyAdapter(config=self.config)
        self.shield_store = AsyncMock()
        self.adapter.shield_store = self.shield_store

        # Mock the HTTP request methods
        self.guardrails_post_patcher = patch(
            "llama_stack.providers.remote.safety.nvidia.nvidia.NeMoGuardrails._guardrails_post"
        )
        self.mock_guardrails_post = self.guardrails_post_patcher.start()
        self.mock_guardrails_post.return_value = {"status": "allowed"}

    def tearDown(self):
        """Clean up after each test."""
        self.guardrails_post_patcher.stop()

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, run_async):
        self.run_async = run_async

    def _assert_request(
        self,
        mock_call: MagicMock,
        expected_url: str,
        expected_headers: dict[str, str] | None = None,
        expected_json: dict[str, Any] | None = None,
    ) -> None:
        """
        Helper method to verify request details in mock API calls.

        Args:
            mock_call: The MagicMock object that was called
            expected_url: The expected URL to which the request was made
            expected_headers: Optional dictionary of expected request headers
            expected_json: Optional dictionary of expected JSON payload
        """
        call_args = mock_call.call_args

        # Check URL
        assert call_args[0][0] == expected_url

        # Check headers if provided
        if expected_headers:
            for key, value in expected_headers.items():
                assert call_args[1]["headers"][key] == value

        # Check JSON if provided
        if expected_json:
            for key, value in expected_json.items():
                if isinstance(value, dict):
                    for nested_key, nested_value in value.items():
                        assert call_args[1]["json"][key][nested_key] == nested_value
                else:
                    assert call_args[1]["json"][key] == value

    def test_register_shield_with_valid_id(self):
        shield = Shield(
            provider_id="nvidia",
            type="shield",
            identifier="test-shield",
            provider_resource_id="test-model",
        )

        # Register the shield
        self.run_async(self.adapter.register_shield(shield))

    def test_register_shield_without_id(self):
        shield = Shield(
            provider_id="nvidia",
            type="shield",
            identifier="test-shield",
            provider_resource_id="",
        )

        # Register the shield should raise a ValueError
        with self.assertRaises(ValueError):
            self.run_async(self.adapter.register_shield(shield))

    def test_run_shield_allowed(self):
        # Set up the shield
        shield_id = "test-shield"
        shield = Shield(
            provider_id="nvidia",
            type="shield",
            identifier=shield_id,
            provider_resource_id="test-model",
        )
        self.shield_store.get_shield.return_value = shield

        # Mock Guardrails API response
        self.mock_guardrails_post.return_value = {"status": "allowed"}

        # Run the shield
        messages = [
            UserMessage(role="user", content="Hello, how are you?"),
            CompletionMessage(
                role="assistant",
                content="I'm doing well, thank you for asking!",
                stop_reason="end_of_message",
                tool_calls=[],
            ),
        ]
        result = self.run_async(self.adapter.run_shield(shield_id, messages))

        # Verify the shield store was called
        self.shield_store.get_shield.assert_called_once_with(shield_id)

        # Verify the Guardrails API was called correctly
        self.mock_guardrails_post.assert_called_once_with(
            path="/v1/guardrail/checks",
            data={
                "model": shield_id,
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
                ],
                "temperature": 1.0,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "max_tokens": 160,
                "stream": False,
                "guardrails": {
                    "config_id": "self-check",
                },
            },
        )

        # Verify the result
        assert isinstance(result, RunShieldResponse)
        assert result.violation is None

    def test_run_shield_blocked(self):
        # Set up the shield
        shield_id = "test-shield"
        shield = Shield(
            provider_id="nvidia",
            type="shield",
            identifier=shield_id,
            provider_resource_id="test-model",
        )
        self.shield_store.get_shield.return_value = shield

        # Mock Guardrails API response
        self.mock_guardrails_post.return_value = {"status": "blocked", "rails_status": {"reason": "harmful_content"}}

        # Run the shield
        messages = [
            UserMessage(role="user", content="Hello, how are you?"),
            CompletionMessage(
                role="assistant",
                content="I'm doing well, thank you for asking!",
                stop_reason="end_of_message",
                tool_calls=[],
            ),
        ]
        result = self.run_async(self.adapter.run_shield(shield_id, messages))

        # Verify the shield store was called
        self.shield_store.get_shield.assert_called_once_with(shield_id)

        # Verify the Guardrails API was called correctly
        self.mock_guardrails_post.assert_called_once_with(
            path="/v1/guardrail/checks",
            data={
                "model": shield_id,
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
                ],
                "temperature": 1.0,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "max_tokens": 160,
                "stream": False,
                "guardrails": {
                    "config_id": "self-check",
                },
            },
        )

        # Verify the result
        assert result.violation is not None
        assert isinstance(result, RunShieldResponse)
        assert result.violation.user_message == "Sorry I cannot do this."
        assert result.violation.violation_level == ViolationLevel.ERROR
        assert result.violation.metadata == {"reason": "harmful_content"}

    def test_run_shield_not_found(self):
        # Set up shield store to return None
        shield_id = "non-existent-shield"
        self.shield_store.get_shield.return_value = None

        messages = [
            UserMessage(role="user", content="Hello, how are you?"),
        ]

        with self.assertRaises(ValueError):
            self.run_async(self.adapter.run_shield(shield_id, messages))

        # Verify the shield store was called
        self.shield_store.get_shield.assert_called_once_with(shield_id)

        # Verify the Guardrails API was not called
        self.mock_guardrails_post.assert_not_called()

    def test_run_shield_http_error(self):
        shield_id = "test-shield"
        shield = Shield(
            provider_id="nvidia",
            type="shield",
            identifier=shield_id,
            provider_resource_id="test-model",
        )
        self.shield_store.get_shield.return_value = shield

        # Mock Guardrails API to raise an exception
        error_msg = "API Error: 500 Internal Server Error"
        self.mock_guardrails_post.side_effect = Exception(error_msg)

        # Running the shield should raise an exception
        messages = [
            UserMessage(role="user", content="Hello, how are you?"),
            CompletionMessage(
                role="assistant",
                content="I'm doing well, thank you for asking!",
                stop_reason="end_of_message",
                tool_calls=[],
            ),
        ]
        with self.assertRaises(Exception) as context:
            self.run_async(self.adapter.run_shield(shield_id, messages))

        # Verify the shield store was called
        self.shield_store.get_shield.assert_called_once_with(shield_id)

        # Verify the Guardrails API was called correctly
        self.mock_guardrails_post.assert_called_once_with(
            path="/v1/guardrail/checks",
            data={
                "model": shield_id,
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
                ],
                "temperature": 1.0,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "max_tokens": 160,
                "stream": False,
                "guardrails": {
                    "config_id": "self-check",
                },
            },
        )
        # Verify the exception message
        assert error_msg in str(context.exception)

    def test_init_nemo_guardrails(self):
        from llama_stack.providers.remote.safety.nvidia.nvidia import NeMoGuardrails

        test_config_id = "test-custom-config-id"
        config = NVIDIASafetyConfig(
            guardrails_service_url=os.environ["NVIDIA_GUARDRAILS_URL"],
            config_id=test_config_id,
        )
        # Initialize with default parameters
        test_model = "test-model"
        guardrails = NeMoGuardrails(config, test_model)

        # Verify the attributes are set correctly
        assert guardrails.config_id == test_config_id
        assert guardrails.model == test_model
        assert guardrails.threshold == 0.9  # Default value
        assert guardrails.temperature == 1.0  # Default value
        assert guardrails.guardrails_service_url == os.environ["NVIDIA_GUARDRAILS_URL"]

        # Initialize with custom parameters
        guardrails = NeMoGuardrails(config, test_model, threshold=0.8, temperature=0.7)

        # Verify the attributes are set correctly
        assert guardrails.config_id == test_config_id
        assert guardrails.model == test_model
        assert guardrails.threshold == 0.8
        assert guardrails.temperature == 0.7
        assert guardrails.guardrails_service_url == os.environ["NVIDIA_GUARDRAILS_URL"]

    def test_init_nemo_guardrails_invalid_temperature(self):
        from llama_stack.providers.remote.safety.nvidia.nvidia import NeMoGuardrails

        config = NVIDIASafetyConfig(
            guardrails_service_url=os.environ["NVIDIA_GUARDRAILS_URL"],
            config_id="test-custom-config-id",
        )
        with self.assertRaises(ValueError):
            NeMoGuardrails(config, "test-model", temperature=0)
