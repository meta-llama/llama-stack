# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from llama_stack.testing.fake_responses import FakeConfig, generate_fake_response, generate_fake_stream


class TestGenerateFakeResponse:
    """Test cases for generate_fake_response function."""

    def test_chat_completions_basic(self):
        """Test basic chat completions generation."""
        endpoint = "/v1/chat/completions"
        body = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello, how are you?"}]}
        config = FakeConfig(response_length=10, latency_ms=50)

        response = generate_fake_response(endpoint, body, config)

        # Check response structure
        if hasattr(response, "id"):
            # OpenAI object format
            assert response.id.startswith("chatcmpl-fake-")
            assert response.object == "chat.completion"
            assert response.model == "gpt-3.5-turbo"
            assert len(response.choices) == 1
            assert response.choices[0].message.role == "assistant"
            assert response.choices[0].message.content is not None
            assert len(response.choices[0].message.content.split()) == 10
            assert response.usage.total_tokens > 0
        else:
            # Dict format fallback
            assert response["id"].startswith("chatcmpl-fake-")
            assert response["object"] == "chat.completion"
            assert response["model"] == "gpt-3.5-turbo"
            assert len(response["choices"]) == 1
            assert response["choices"][0]["message"]["role"] == "assistant"
            assert response["choices"][0]["message"]["content"] is not None
            assert len(response["choices"][0]["message"]["content"].split()) == 10
            assert response["usage"]["total_tokens"] > 0

    def test_chat_completions_custom_model(self):
        """Test chat completions with custom model name."""
        endpoint = "/v1/chat/completions"
        body = {"model": "custom-model-name", "messages": [{"role": "user", "content": "Test message"}]}
        config = FakeConfig(response_length=5, latency_ms=10)

        response = generate_fake_response(endpoint, body, config)

        # Check model name is preserved
        if hasattr(response, "model"):
            assert response.model == "custom-model-name"
        else:
            assert response["model"] == "custom-model-name"

    def test_chat_completions_multiple_messages(self):
        """Test chat completions with multiple input messages."""
        endpoint = "/v1/chat/completions"
        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you doing today?"},
            ],
        }
        config = FakeConfig(response_length=15, latency_ms=25)

        response = generate_fake_response(endpoint, body, config)

        # Check token calculation includes all messages
        if hasattr(response, "usage"):
            assert response.usage.prompt_tokens > 0  # Should count all input messages
            assert response.usage.completion_tokens == 15
        else:
            assert response["usage"]["prompt_tokens"] > 0
            assert response["usage"]["completion_tokens"] == 15

    def test_completions_not_implemented(self):
        """Test that completions endpoint raises NotImplementedError."""
        endpoint = "/v1/completions"
        body = {"model": "gpt-3.5-turbo-instruct", "prompt": "Test prompt"}
        config = FakeConfig(response_length=10)

        with pytest.raises(NotImplementedError, match="Fake completions not implemented yet"):
            generate_fake_response(endpoint, body, config)

    def test_embeddings_not_implemented(self):
        """Test that embeddings endpoint raises NotImplementedError."""
        endpoint = "/v1/embeddings"
        body = {"model": "text-embedding-ada-002", "input": "Test text"}
        config = FakeConfig()

        with pytest.raises(NotImplementedError, match="Fake embeddings not implemented yet"):
            generate_fake_response(endpoint, body, config)

    def test_models_not_implemented(self):
        """Test that models endpoint raises NotImplementedError."""
        endpoint = "/v1/models"
        body = {}
        config = FakeConfig()

        with pytest.raises(NotImplementedError, match="Fake models list not implemented yet"):
            generate_fake_response(endpoint, body, config)

    def test_unsupported_endpoint(self):
        """Test that unsupported endpoints raise ValueError."""
        endpoint = "/v1/unknown"
        body = {}
        config = FakeConfig()

        with pytest.raises(ValueError, match="Unsupported endpoint for fake mode: /v1/unknown"):
            generate_fake_response(endpoint, body, config)

    def test_content_with_arrays(self):
        """Test chat completions with content arrays (e.g., images)."""
        endpoint = "/v1/chat/completions"
        body = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
                    ],
                }
            ],
        }
        config = FakeConfig(response_length=20)

        response = generate_fake_response(endpoint, body, config)

        # Should handle content arrays without errors
        if hasattr(response, "usage"):
            assert response.usage.prompt_tokens > 0
        else:
            assert response["usage"]["prompt_tokens"] > 0


class TestGenerateFakeStream:
    """Test cases for generate_fake_stream function."""

    @pytest.mark.asyncio
    async def test_chat_completions_streaming(self):
        """Test streaming chat completions generation."""
        # First generate a response
        response_data = generate_fake_response(
            "/v1/chat/completions",
            {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello"}]},
            FakeConfig(response_length=5, latency_ms=1),  # Very low latency for testing
        )

        # Then stream it
        chunks = []
        async for chunk in generate_fake_stream(response_data, "/v1/chat/completions", FakeConfig(latency_ms=1)):
            chunks.append(chunk)

        # Should have initial role chunk + content chunks + final chunk
        assert len(chunks) >= 3

        # First chunk should have role
        first_chunk = chunks[0]
        assert first_chunk["object"] == "chat.completion.chunk"
        assert first_chunk["choices"][0]["delta"]["role"] == "assistant"
        assert first_chunk["choices"][0]["delta"]["content"] == ""

        # Middle chunks should have content
        content_chunks = [c for c in chunks[1:-1] if c["choices"][0]["delta"].get("content")]
        assert len(content_chunks) > 0

        # Last chunk should have finish_reason
        last_chunk = chunks[-1]
        assert last_chunk["choices"][0]["finish_reason"] == "stop"
        assert last_chunk["choices"][0]["delta"]["content"] is None

    @pytest.mark.asyncio
    async def test_completions_streaming_not_implemented(self):
        """Test that streaming completions raises NotImplementedError."""
        response_data = {"id": "test", "choices": [{"text": "test content"}]}

        stream = generate_fake_stream(response_data, "/v1/completions", FakeConfig())

        with pytest.raises(NotImplementedError, match="Fake streaming completions not implemented yet"):
            async for _ in stream:
                pass
