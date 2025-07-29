# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from openai import AsyncOpenAI

from llama_stack.testing.inference_recorder import (
    ResponseStorage,
    inference_recording,
    normalize_request,
)


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for test recordings."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI response object."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Hello! I'm doing well, thank you for asking."
    mock_response.model_dump.return_value = {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello! I'm doing well, thank you for asking."},
                "finish_reason": "stop",
            }
        ],
        "model": "llama3.2:3b",
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }

    return mock_response


@pytest.fixture
def mock_embeddings_response():
    """Mock OpenAI embeddings response object."""
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3]), Mock(embedding=[0.4, 0.5, 0.6])]
    mock_response.model_dump.return_value = {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
            {"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 1},
        ],
        "model": "nomic-embed-text",
        "usage": {"prompt_tokens": 6, "total_tokens": 6},
    }

    return mock_response


class TestInferenceRecording:
    """Test the inference recording system."""

    def test_request_normalization(self):
        """Test that request normalization produces consistent hashes."""
        # Test basic normalization
        hash1 = normalize_request(
            "POST",
            "http://localhost:11434/v1/chat/completions",
            {},
            {"model": "llama3.2:3b", "messages": [{"role": "user", "content": "Hello world"}], "temperature": 0.7},
        )

        # Same request should produce same hash
        hash2 = normalize_request(
            "POST",
            "http://localhost:11434/v1/chat/completions",
            {},
            {"model": "llama3.2:3b", "messages": [{"role": "user", "content": "Hello world"}], "temperature": 0.7},
        )

        assert hash1 == hash2

        # Different content should produce different hash
        hash3 = normalize_request(
            "POST",
            "http://localhost:11434/v1/chat/completions",
            {},
            {
                "model": "llama3.2:3b",
                "messages": [{"role": "user", "content": "Different message"}],
                "temperature": 0.7,
            },
        )

        assert hash1 != hash3

    def test_request_normalization_edge_cases(self):
        """Test request normalization handles edge cases correctly."""
        # Test whitespace normalization
        hash1 = normalize_request(
            "POST",
            "http://test/v1/chat/completions",
            {},
            {"messages": [{"role": "user", "content": "Hello   world\n\n"}]},
        )
        hash2 = normalize_request(
            "POST", "http://test/v1/chat/completions", {}, {"messages": [{"role": "user", "content": "Hello world"}]}
        )
        assert hash1 == hash2

        # Test float precision normalization
        hash3 = normalize_request("POST", "http://test/v1/chat/completions", {}, {"temperature": 0.7000001})
        hash4 = normalize_request("POST", "http://test/v1/chat/completions", {}, {"temperature": 0.7})
        assert hash3 == hash4

    def test_response_storage(self, temp_storage_dir):
        """Test the ResponseStorage class."""
        storage = ResponseStorage(temp_storage_dir, "test_storage")

        # Test directory creation
        assert storage.test_dir.exists()
        assert storage.responses_dir.exists()
        assert storage.db_path.exists()

        # Test storing and retrieving a recording
        request_hash = "test_hash_123"
        request_data = {
            "method": "POST",
            "url": "http://localhost:11434/v1/chat/completions",
            "endpoint": "/v1/chat/completions",
            "model": "llama3.2:3b",
        }
        response_data = {"body": {"content": "test response"}, "is_streaming": False}

        storage.store_recording(request_hash, request_data, response_data)

        # Verify SQLite record
        with sqlite3.connect(storage.db_path) as conn:
            result = conn.execute("SELECT * FROM recordings WHERE request_hash = ?", (request_hash,)).fetchone()

        assert result is not None
        assert result[0] == request_hash  # request_hash
        assert result[2] == "/v1/chat/completions"  # endpoint
        assert result[3] == "llama3.2:3b"  # model

        # Verify file storage and retrieval
        retrieved = storage.find_recording(request_hash)
        assert retrieved is not None
        assert retrieved["request"]["model"] == "llama3.2:3b"
        assert retrieved["response"]["body"]["content"] == "test response"

    async def test_recording_mode(self, temp_storage_dir, mock_openai_response):
        """Test that recording mode captures and stores responses."""
        test_id = "test_recording_mode"

        async def mock_create(*args, **kwargs):
            return mock_openai_response

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_create):
            with inference_recording(mode="record", test_id=test_id, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                response = await client.chat.completions.create(
                    model="llama3.2:3b",
                    messages=[{"role": "user", "content": "Hello, how are you?"}],
                    temperature=0.7,
                    max_tokens=50,
                )

                # Verify the response was returned correctly
                assert response.choices[0].message.content == "Hello! I'm doing well, thank you for asking."

        # Verify recording was stored
        storage = ResponseStorage(temp_storage_dir, test_id)
        with sqlite3.connect(storage.db_path) as conn:
            recordings = conn.execute("SELECT COUNT(*) FROM recordings").fetchone()[0]

        assert recordings == 1

    async def test_replay_mode(self, temp_storage_dir, mock_openai_response):
        """Test that replay mode returns stored responses without making real calls."""
        test_id = "test_replay_mode"

        async def mock_create(*args, **kwargs):
            return mock_openai_response

        # First, record a response
        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_create):
            with inference_recording(mode="record", test_id=test_id, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                await client.chat.completions.create(
                    model="llama3.2:3b",
                    messages=[{"role": "user", "content": "Hello, how are you?"}],
                    temperature=0.7,
                    max_tokens=50,
                )

        # Now test replay mode - should not call the original method
        with patch("openai.resources.chat.completions.AsyncCompletions.create") as mock_create_patch:
            with inference_recording(mode="replay", test_id=test_id, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                response = await client.chat.completions.create(
                    model="llama3.2:3b",
                    messages=[{"role": "user", "content": "Hello, how are you?"}],
                    temperature=0.7,
                    max_tokens=50,
                )

                # Verify we got the recorded response
                assert response["choices"][0]["message"]["content"] == "Hello! I'm doing well, thank you for asking."

                # Verify the original method was NOT called
                mock_create_patch.assert_not_called()

    async def test_replay_missing_recording(self, temp_storage_dir):
        """Test that replay mode fails when no recording is found."""
        test_id = "test_missing_recording"

        with patch("openai.resources.chat.completions.AsyncCompletions.create"):
            with inference_recording(mode="replay", test_id=test_id, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                with pytest.raises(RuntimeError, match="No recorded response found"):
                    await client.chat.completions.create(
                        model="llama3.2:3b", messages=[{"role": "user", "content": "This was never recorded"}]
                    )

    async def test_embeddings_recording(self, temp_storage_dir, mock_embeddings_response):
        """Test recording and replay of embeddings calls."""
        test_id = "test_embeddings"

        async def mock_create(*args, **kwargs):
            return mock_embeddings_response

        # Record
        with patch("openai.resources.embeddings.AsyncEmbeddings.create", side_effect=mock_create):
            with inference_recording(mode="record", test_id=test_id, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                response = await client.embeddings.create(
                    model="nomic-embed-text", input=["Hello world", "Test embedding"]
                )

                assert len(response.data) == 2

        # Replay
        with patch("openai.resources.embeddings.AsyncEmbeddings.create") as mock_create_patch:
            with inference_recording(mode="replay", test_id=test_id, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                response = await client.embeddings.create(
                    model="nomic-embed-text", input=["Hello world", "Test embedding"]
                )

                # Verify we got the recorded response
                assert len(response["data"]) == 2
                assert response["data"][0]["embedding"] == [0.1, 0.2, 0.3]

                # Verify original method was not called
                mock_create_patch.assert_not_called()

    async def test_live_mode(self, mock_openai_response):
        """Test that live mode passes through to original methods."""

        async def mock_create(*args, **kwargs):
            return mock_openai_response

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_create):
            with inference_recording(mode="live"):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                response = await client.chat.completions.create(
                    model="llama3.2:3b", messages=[{"role": "user", "content": "Hello"}]
                )

                # Verify the response was returned
                assert response.choices[0].message.content == "Hello! I'm doing well, thank you for asking."
