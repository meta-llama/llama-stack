# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import Mock, patch

import pytest

from llama_stack.apis.inference import StopReason, ToolCall
from llama_stack.models.llama.datatypes import RawMessage
from llama_stack.models.llama.sku_types import ModelFamily
from llama_stack.providers.utils.inference.prompt_adapter import decode_assistant_message


@pytest.fixture
def test_content():
    """Test content fixtures."""
    return {
        "llama3_content": """I'll help you get the weather information.

{"type": "function", "name": "get_weather", "parameters": {"location": "San Francisco, CA"}}""",
        "llama4_content": """I'll help you get the weather information.

[get_weather(location="San Francisco, CA")]""",
        "simple_content": "Hello! How can I help you today?",
    }


def test_decode_with_no_model_id_defaults_to_llama3(test_content):
    """Test that decode_assistant_message defaults to Llama3 format when no model_id is provided."""
    with patch("llama_stack.providers.utils.inference.prompt_adapter.ChatFormat") as mock_chat_format:
        mock_formatter = Mock()
        mock_chat_format.return_value = mock_formatter
        expected_message = RawMessage(role="assistant", content=test_content["simple_content"])
        mock_formatter.decode_assistant_message_from_content.return_value = expected_message

        result = decode_assistant_message(test_content["simple_content"], StopReason.end_of_turn)

        mock_chat_format.assert_called_once()
        mock_formatter.decode_assistant_message_from_content.assert_called_once_with(
            test_content["simple_content"], StopReason.end_of_turn
        )
        assert result == expected_message


@patch("llama_stack.providers.utils.inference.prompt_adapter.LLAMA4_AVAILABLE", True)
def test_decode_with_nonexistent_model_uses_llama3(test_content):
    """Test that decode_assistant_message uses Llama3 format for non-existent models."""
    with patch("llama_stack.providers.utils.inference.prompt_adapter.resolve_model") as mock_resolve:
        mock_resolve.return_value = None

        with patch("llama_stack.providers.utils.inference.prompt_adapter.ChatFormat") as mock_chat_format:
            mock_formatter = Mock()
            mock_chat_format.return_value = mock_formatter
            expected_message = RawMessage(role="assistant", content=test_content["simple_content"])
            mock_formatter.decode_assistant_message_from_content.return_value = expected_message

            result = decode_assistant_message(
                test_content["simple_content"], StopReason.end_of_turn, "nonexistent-model"
            )

            mock_resolve.assert_called_once_with("nonexistent-model")
            mock_chat_format.assert_called_once()
            assert result == expected_message


@patch("llama_stack.providers.utils.inference.prompt_adapter.LLAMA4_AVAILABLE", True)
def test_decode_with_llama3_model_uses_llama3_format(test_content):
    """Test that decode_assistant_message uses Llama3 format for Llama3 models."""
    mock_model = Mock()
    mock_model.model_family = ModelFamily.llama3

    with patch("llama_stack.providers.utils.inference.prompt_adapter.resolve_model") as mock_resolve:
        mock_resolve.return_value = mock_model

        with patch("llama_stack.providers.utils.inference.prompt_adapter.ChatFormat") as mock_chat_format:
            mock_formatter = Mock()
            mock_chat_format.return_value = mock_formatter
            expected_message = RawMessage(role="assistant", content=test_content["llama3_content"])
            mock_formatter.decode_assistant_message_from_content.return_value = expected_message

            result = decode_assistant_message(
                test_content["llama3_content"], StopReason.end_of_turn, "meta-llama/Llama-3.1-8B-Instruct"
            )

            mock_resolve.assert_called_once_with("meta-llama/Llama-3.1-8B-Instruct")
            mock_chat_format.assert_called_once()
            assert result == expected_message


@patch("llama_stack.providers.utils.inference.prompt_adapter.LLAMA4_AVAILABLE", True)
def test_decode_with_llama4_model_uses_llama4_format(test_content):
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
                expected_message = RawMessage(role="assistant", content=test_content["llama4_content"])
                mock_formatter.decode_assistant_message_from_content.return_value = expected_message

                result = decode_assistant_message(
                    test_content["llama4_content"], StopReason.end_of_turn, "meta-llama/Llama-4-8B-Instruct"
                )

                mock_resolve.assert_called_once_with("meta-llama/Llama-4-8B-Instruct")
                mock_llama4_format.assert_called_once_with(mock_tokenizer_instance)
                assert result == expected_message


@patch("llama_stack.providers.utils.inference.prompt_adapter.LLAMA4_AVAILABLE", False)
def test_decode_with_llama4_model_falls_back_to_llama3_when_unavailable(test_content):
    """Test that decode_assistant_message falls back to Llama3 format when Llama4 is unavailable."""
    mock_model = Mock()
    mock_model.model_family = ModelFamily.llama4

    with patch("llama_stack.providers.utils.inference.prompt_adapter.resolve_model") as mock_resolve:
        mock_resolve.return_value = mock_model

        with patch("llama_stack.providers.utils.inference.prompt_adapter.ChatFormat") as mock_chat_format:
            mock_formatter = Mock()
            mock_chat_format.return_value = mock_formatter
            expected_message = RawMessage(role="assistant", content=test_content["llama4_content"])
            mock_formatter.decode_assistant_message_from_content.return_value = expected_message

            result = decode_assistant_message(
                test_content["llama4_content"], StopReason.end_of_turn, "meta-llama/Llama-4-8B-Instruct"
            )

            # Should NOT call resolve_model since LLAMA4_AVAILABLE is False
            mock_resolve.assert_not_called()
            mock_chat_format.assert_called_once()
            assert result == expected_message


@pytest.mark.parametrize(
    "stop_reason",
    [
        StopReason.end_of_turn,
        StopReason.end_of_message,
        StopReason.out_of_tokens,
    ],
)
def test_decode_with_different_stop_reasons(test_content, stop_reason):
    """Test that decode_assistant_message handles different stop reasons correctly."""
    with patch("llama_stack.providers.utils.inference.prompt_adapter.ChatFormat") as mock_chat_format:
        mock_formatter = Mock()
        mock_chat_format.return_value = mock_formatter
        expected_message = RawMessage(role="assistant", content=test_content["simple_content"])
        mock_formatter.decode_assistant_message_from_content.return_value = expected_message

        result = decode_assistant_message(test_content["simple_content"], stop_reason)

        mock_formatter.decode_assistant_message_from_content.assert_called_once_with(
            test_content["simple_content"], stop_reason
        )
        assert result == expected_message


def test_decode_preserves_formatter_response(test_content):
    """Test that decode_assistant_message preserves the formatter's response including tool calls."""
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

        result = decode_assistant_message(test_content["llama3_content"], StopReason.end_of_turn)

        assert result == expected_message
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "get_weather"


@pytest.mark.parametrize(
    "model_id,description",
    [
        ("meta-llama/Llama-3.1-8B-Instruct", "should resolve to Llama3"),
        ("meta-llama/Llama-4-8B-Instruct", "should resolve to Llama4 if available"),
        ("invalid-model-id", "should fallback to Llama3"),
    ],
)
def test_model_resolution_integration(model_id, description):
    """Test that model resolution works correctly with actual model IDs."""
    with patch("llama_stack.providers.utils.inference.prompt_adapter.ChatFormat") as mock_chat_format:
        mock_formatter = Mock()
        mock_chat_format.return_value = mock_formatter
        expected_message = RawMessage(role="assistant", content="Test content")
        mock_formatter.decode_assistant_message_from_content.return_value = expected_message

        # This should not raise an exception
        result = decode_assistant_message("Test content", StopReason.end_of_turn, model_id)

        assert result == expected_message
