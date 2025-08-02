# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Unit tests for LlamaStackAsLibraryClient automatic initialization.

These tests ensure that the library client is automatically initialized
and ready to use immediately after construction.
"""

from llama_stack.core.library_client import (
    AsyncLlamaStackAsLibraryClient,
    LlamaStackAsLibraryClient,
)


class TestLlamaStackAsLibraryClientAutoInitialization:
    """Test automatic initialization of library clients."""

    def test_sync_client_auto_initialization(self):
        """Test that sync client is automatically initialized after construction."""
        client = LlamaStackAsLibraryClient("nvidia")

        # Client should be automatically initialized
        assert client.async_client._is_initialized is True
        assert client.async_client.route_impls is not None

    async def test_async_client_auto_initialization(self):
        """Test that async client can be initialized and works properly."""
        client = AsyncLlamaStackAsLibraryClient("nvidia")

        # Initialize the client
        result = await client.initialize()
        assert result is True
        assert client._is_initialized is True
        assert client.route_impls is not None

    def test_initialize_method_backward_compatibility(self):
        """Test that initialize() method still works for backward compatibility."""
        client = LlamaStackAsLibraryClient("nvidia")

        # initialize() should return None (historical behavior) and not cause errors
        result = client.initialize()
        assert result is None

        # Multiple calls should be safe
        result2 = client.initialize()
        assert result2 is None

    async def test_async_initialize_method_idempotent(self):
        """Test that async initialize() method can be called multiple times safely."""
        client = AsyncLlamaStackAsLibraryClient("nvidia")

        # First initialization
        result1 = await client.initialize()
        assert result1 is True
        assert client._is_initialized is True

        # Second initialization should be safe and return True
        result2 = await client.initialize()
        assert result2 is True
        assert client._is_initialized is True

    def test_route_impls_automatically_set(self):
        """Test that route_impls is automatically set during construction."""
        # Test sync client - should be auto-initialized
        sync_client = LlamaStackAsLibraryClient("nvidia")
        assert sync_client.async_client.route_impls is not None

        # Test that the async client is marked as initialized
        assert sync_client.async_client._is_initialized is True
