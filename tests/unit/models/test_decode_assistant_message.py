# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import unittest
from unittest.mock import Mock, patch

from llama_stack.apis.inference import StopReason
from llama_stack.models.llama.datatypes import RawMessage
from llama_stack.models.llama.sku_types import ModelFamily
from llama_stack.providers.utils.inference.prompt_adapter import decode_assistant_message


class TestDecodeAssistantMessage(unittest.TestCase):
    """Test the decode_assistant_message function with different model types and formats."""

    def setUp(self):
        """Set up test fixtures."""
        self.llama3_content = """I'll help you get the weather information.

{"type": "function", "name": "get_weather", "parameters": {"location": "San Francisco, CA"}}"""

        self.llama4_content = """I'll help you get the weather information.

[get_weather(location="San Francisco, CA")]"""

        self.simple_content = "Hello! How can I help you today?"

    def test_decode_with_no_model_id_defaults_to_llama3(self):
        """Test that decode_assistant_message defaults to Llama3 format when no model_id is provided."""
        with patch("llama_stack.providers.utils.inference.prompt_adapter.ChatFormat") as mock_chat_format:
            mock_formatter = Mock()
            mock_chat_format.return_value = mock_formatter
            expected_message = RawMessage(role="assistant", content=self.simple_content)
            mock_formatter.decode_assistant_message_from_content.return_value = expected_message

            result = decode_assistant_message(self.simple_content, StopReason.end_of_turn)

            mock_chat_format.assert_called_once()
            mock_formatter.decode_assistant_message_from_content.assert_called_once_with(
                self.simple_content, StopReason.end_of_turn
            )
            self.assertEqual(result, expected_message)

    @patch("llama_stack.providers.utils.inference.prompt_adapter.LLAMA4_AVAILABLE", True)
    def test_decode_with_nonexistent_model_uses_llama3(self):
        """Test that decode_assistant_message uses Llama3 format for non-existent models."""
        with patch("llama_stack.providers.utils.inference.prompt_adapter.resolve_model") as mock_resolve:
            mock_resolve.return_value = None

            with patch("llama_stack.providers.utils.inference.prompt_adapter.ChatFormat") as mock_chat_format:
                mock_formatter = Mock()
                mock_chat_format.return_value = mock_formatter
                expected_message = RawMessage(role="assistant", content=self.simple_content)
                mock_formatter.decode_assistant_message_from_content.return_value = expected_message

                result = decode_assistant_message(self.simple_content, StopReason.end_of_turn, "nonexistent-model")

                mock_resolve.assert_called_once_with("nonexistent-model")
                mock_chat_format.assert_called_once()
                self.assertEqual(result, expected_message)

    @patch("llama_stack.providers.utils.inference.prompt_adapter.LLAMA4_AVAILABLE", True)
    def test_decode_with_llama3_model_uses_llama3_format(self):
        """Test that decode_assistant_message uses Llama3 format for Llama3 models."""
        mock_model = Mock()
        mock_model.model_family = ModelFamily.llama3

        with patch("llama_stack.providers.utils.inference.prompt_adapter.resolve_model") as mock_resolve:
            mock_resolve.return_value = mock_model

            with patch("llama_stack.providers.utils.inference.prompt_adapter.ChatFormat") as mock_chat_format:
                mock_formatter = Mock()
                mock_chat_format.return_value = mock_formatter
                expected_message = RawMessage(role="assistant", content=self.llama3_content)
                mock_formatter.decode_assistant_message_from_content.return_value = expected_message

                result = decode_assistant_message(
                    self.llama3_content, StopReason.end_of_turn, "meta-llama/Llama-3.1-8B-Instruct"
                )

                mock_resolve.assert_called_once_with("meta-llama/Llama-3.1-8B-Instruct")
                mock_chat_format.assert_called_once()
                self.assertEqual(result, expected_message)

    @patch("llama_stack.providers.utils.inference.prompt_adapter.LLAMA4_AVAILABLE", True)
    def test_decode_with_llama4_model_uses_llama4_format(self):
        """Test that decode_assistant_message uses Llama4 format for Llama4 models when available."""
        mock_model = Mock()
        mock_model.model_family = ModelFamily.llama4

        with patch("llama_stack.providers.utils.inference.prompt_adapter.resolve_model") as mock_resolve:
            mock_resolve.return_value = mock_model

            # Mock the Llama4 components
            with patch(
                "llama_stack.providers.utils.inference.prompt_adapter.Llama4ChatFormat", create=True
            ) as mock_llama4_format:
                with patch(
                    "llama_stack.providers.utils.inference.prompt_adapter.Llama4Tokenizer", create=True
                ) as mock_llama4_tokenizer:
                    mock_tokenizer_instance = Mock()
                    mock_llama4_tokenizer.get_instance.return_value = mock_tokenizer_instance

                    mock_formatter = Mock()
                    mock_llama4_format.return_value = mock_formatter
                    expected_message = RawMessage(role="assistant", content=self.llama4_content)
                    mock_formatter.decode_assistant_message_from_content.return_value = expected_message

                    result = decode_assistant_message(
                        self.llama4_content, StopReason.end_of_turn, "meta-llama/Llama-4-8B-Instruct"
                    )

                    mock_resolve.assert_called_once_with("meta-llama/Llama-4-8B-Instruct")
                    mock_llama4_format.assert_called_once_with(mock_tokenizer_instance)
                    self.assertEqual(result, expected_message)

    @patch("llama_stack.providers.utils.inference.prompt_adapter.LLAMA4_AVAILABLE", False)
    def test_decode_with_llama4_model_falls_back_to_llama3_when_unavailable(self):
        """Test that decode_assistant_message falls back to Llama3 format when Llama4 is unavailable."""
        mock_model = Mock()
        mock_model.model_family = ModelFamily.llama4

        with patch("llama_stack.providers.utils.inference.prompt_adapter.resolve_model") as mock_resolve:
            mock_resolve.return_value = mock_model

            with patch("llama_stack.providers.utils.inference.prompt_adapter.ChatFormat") as mock_chat_format:
                mock_formatter = Mock()
                mock_chat_format.return_value = mock_formatter
                expected_message = RawMessage(role="assistant", content=self.llama4_content)
                mock_formatter.decode_assistant_message_from_content.return_value = expected_message

                result = decode_assistant_message(
                    self.llama4_content, StopReason.end_of_turn, "meta-llama/Llama-4-8B-Instruct"
                )

                # Should NOT call resolve_model since LLAMA4_AVAILABLE is False
                mock_resolve.assert_not_called()
                mock_chat_format.assert_called_once()
                self.assertEqual(result, expected_message)

    def test_decode_with_different_stop_reasons(self):
        """Test that decode_assistant_message handles different stop reasons correctly."""
        stop_reasons = [
            StopReason.end_of_turn,
            StopReason.end_of_message,
            StopReason.out_of_tokens,
        ]

        for stop_reason in stop_reasons:
            with self.subTest(stop_reason=stop_reason):
                with patch("llama_stack.providers.utils.inference.prompt_adapter.ChatFormat") as mock_chat_format:
                    mock_formatter = Mock()
                    mock_chat_format.return_value = mock_formatter
                    expected_message = RawMessage(role="assistant", content=self.simple_content)
                    mock_formatter.decode_assistant_message_from_content.return_value = expected_message

                    result = decode_assistant_message(self.simple_content, stop_reason)

                    mock_formatter.decode_assistant_message_from_content.assert_called_once_with(
                        self.simple_content, stop_reason
                    )
                    self.assertEqual(result, expected_message)

    def test_decode_preserves_formatter_response(self):
        """Test that decode_assistant_message preserves the formatter's response including tool calls."""
        from llama_stack.apis.inference import ToolCall

        mock_tool_call = ToolCall(
            tool_name="get_weather", arguments={"location": "San Francisco, CA"}, call_id="test_call_id"
        )

        with patch("llama_stack.providers.utils.inference.prompt_adapter.ChatFormat") as mock_chat_format:
            mock_formatter = Mock()
            mock_chat_format.return_value = mock_formatter
            expected_message = RawMessage(
                role="assistant", content="I'll help you get the weather.", tool_calls=[mock_tool_call]
            )
            mock_formatter.decode_assistant_message_from_content.return_value = expected_message

            result = decode_assistant_message(self.llama3_content, StopReason.end_of_turn)

            self.assertEqual(result, expected_message)
            self.assertEqual(len(result.tool_calls), 1)
            self.assertEqual(result.tool_calls[0].tool_name, "get_weather")


class TestDecodeAssistantMessageIntegration(unittest.TestCase):
    """Integration tests for decode_assistant_message with real model resolution."""

    def test_model_resolution_integration(self):
        """Test that model resolution works correctly with actual model IDs."""
        # Test with actual model IDs that should resolve
        test_cases = [
            ("meta-llama/Llama-3.1-8B-Instruct", "should resolve to Llama3"),
            ("meta-llama/Llama-4-8B-Instruct", "should resolve to Llama4 if available"),
            ("invalid-model-id", "should fallback to Llama3"),
        ]

        for model_id, description in test_cases:
            with self.subTest(model_id=model_id, description=description):
                with patch("llama_stack.providers.utils.inference.prompt_adapter.ChatFormat") as mock_chat_format:
                    mock_formatter = Mock()
                    mock_chat_format.return_value = mock_formatter
                    expected_message = RawMessage(role="assistant", content="Test content")
                    mock_formatter.decode_assistant_message_from_content.return_value = expected_message

                    # This should not raise an exception
                    result = decode_assistant_message("Test content", StopReason.end_of_turn, model_id)

                    self.assertEqual(result, expected_message)


if __name__ == "__main__":
    unittest.main()
