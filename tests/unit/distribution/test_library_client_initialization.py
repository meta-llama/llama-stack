# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Unit tests for LlamaStackAsLibraryClient initialization error handling.

These tests ensure that users get proper error messages when they forget to call
initialize() on the library client, preventing AttributeError regressions.
"""

import pytest

from llama_stack.core.library_client import (
    AsyncLlamaStackAsLibraryClient,
    LlamaStackAsLibraryClient,
)


class TestLlamaStackAsLibraryClientInitialization:
    """Test proper error handling for uninitialized library clients."""

    @pytest.mark.parametrize(
        "api_call",
        [
            lambda client: client.models.list(),
            lambda client: client.chat.completions.create(model="test", messages=[{"role": "user", "content": "test"}]),
            lambda client: next(
                client.chat.completions.create(
                    model="test", messages=[{"role": "user", "content": "test"}], stream=True
                )
            ),
        ],
        ids=["models.list", "chat.completions.create", "chat.completions.create_stream"],
    )
    def test_sync_client_proper_error_without_initialization(self, api_call):
        """Test that sync client raises ValueError with helpful message when not initialized."""
        client = LlamaStackAsLibraryClient("nvidia")

        with pytest.raises(ValueError) as exc_info:
            api_call(client)

        error_msg = str(exc_info.value)
        assert "Client not initialized" in error_msg
        assert "Please call initialize() first" in error_msg

    @pytest.mark.parametrize(
        "api_call",
        [
            lambda client: client.models.list(),
            lambda client: client.chat.completions.create(model="test", messages=[{"role": "user", "content": "test"}]),
        ],
        ids=["models.list", "chat.completions.create"],
    )
    async def test_async_client_proper_error_without_initialization(self, api_call):
        """Test that async client raises ValueError with helpful message when not initialized."""
        client = AsyncLlamaStackAsLibraryClient("nvidia")

        with pytest.raises(ValueError) as exc_info:
            await api_call(client)

        error_msg = str(exc_info.value)
        assert "Client not initialized" in error_msg
        assert "Please call initialize() first" in error_msg

    async def test_async_client_streaming_error_without_initialization(self):
        """Test that async client streaming raises ValueError with helpful message when not initialized."""
        client = AsyncLlamaStackAsLibraryClient("nvidia")

        with pytest.raises(ValueError) as exc_info:
            stream = await client.chat.completions.create(
                model="test", messages=[{"role": "user", "content": "test"}], stream=True
            )
            await anext(stream)

        error_msg = str(exc_info.value)
        assert "Client not initialized" in error_msg
        assert "Please call initialize() first" in error_msg

    def test_route_impls_initialized_to_none(self):
        """Test that route_impls is initialized to None to prevent AttributeError."""
        # Test sync client
        sync_client = LlamaStackAsLibraryClient("nvidia")
        assert sync_client.async_client.route_impls is None

        # Test async client directly
        async_client = AsyncLlamaStackAsLibraryClient("nvidia")
        assert async_client.route_impls is None
