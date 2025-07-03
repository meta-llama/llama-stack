# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest
from pydantic import ValidationError

from llama_stack.apis.providers import ProviderInfo
from llama_stack.distribution.datatypes import Provider, StackRunConfig
from llama_stack.distribution.providers import ProviderImpl, ProviderImplConfig


class TestProviderImpl:
    """Test suite for ProviderImpl class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        run_config = StackRunConfig(
            image_name="test_image",
            providers={
                "inference": [
                    Provider(
                        provider_id="test_provider_with_metrics_url",
                        provider_type="test_type1",
                        config={"url": "http://localhost:8000", "metrics": "http://localhost:9090/metrics"},
                    ),
                    Provider(
                        provider_id="test_provider_no_metrics_url",
                        provider_type="test_type2",
                        config={"url": "http://localhost:8080"},
                    ),
                ]
            },
        )
        return ProviderImplConfig(run_config=run_config)

    @pytest.fixture
    def mock_config_malformed_metrics(self):
        """Create a mock configuration with invalid metrics URL for testing."""
        run_config = StackRunConfig(
            image_name="test_image",
            providers={
                "inference": [
                    Provider(
                        provider_id="test_provider_malformed_metrics",
                        provider_type="test_type3",
                        config={"url": "http://localhost:8000", "metrics": "abcde-llama-stack"},
                    ),
                ]
            },
        )
        return ProviderImplConfig(run_config=run_config)

    @pytest.fixture
    def mock_deps(self):
        """Create mock dependencies."""
        return {}

    @pytest.mark.asyncio
    async def test_provider_info_structure(self, mock_config, mock_deps):
        """Test ProviderInfo objects"""
        provider_impl = ProviderImpl(mock_config, mock_deps)

        response = await provider_impl.list_providers()
        provider = response.data[0]

        # Check all required fields
        assert hasattr(provider, "api")
        assert isinstance(provider.api, str)

        assert hasattr(provider, "provider_id")
        assert isinstance(provider.provider_id, str)

        assert hasattr(provider, "provider_type")
        assert isinstance(provider.provider_type, str)

        assert hasattr(provider, "config")
        assert isinstance(provider.config, dict)

        assert hasattr(provider, "health")

        assert provider.metrics is None or isinstance(provider.metrics, str)

    @pytest.mark.asyncio
    async def test_list_providers_with_metrics(self, mock_config, mock_deps):
        """Test list_providers includes metrics field."""
        provider_impl = ProviderImpl(mock_config, mock_deps)

        response = await provider_impl.list_providers()

        assert response is not None
        assert len(response.data) == 2

        # Check provider with metrics
        provider1 = response.data[0]
        assert isinstance(provider1, ProviderInfo)
        assert provider1.provider_id == "test_provider_with_metrics_url"
        assert provider1.metrics == "http://localhost:9090/metrics"

        # Check provider without metrics
        provider2 = response.data[1]
        assert isinstance(provider2, ProviderInfo)
        assert provider2.provider_id == "test_provider_no_metrics_url"
        assert provider2.metrics is None

    @pytest.mark.asyncio
    async def test_inspect_provider_with_metrics(self, mock_config, mock_deps):
        """Test inspect_provider includes metrics field."""
        provider_impl = ProviderImpl(mock_config, mock_deps)

        # Test provider with metrics
        provider_info = await provider_impl.inspect_provider("test_provider_with_metrics_url")
        assert provider_info.provider_id == "test_provider_with_metrics_url"
        assert provider_info.metrics == "http://localhost:9090/metrics"

        # Test provider without metrics
        provider_info = await provider_impl.inspect_provider("test_provider_no_metrics_url")
        assert provider_info.provider_id == "test_provider_no_metrics_url"
        assert provider_info.metrics is None

    @pytest.mark.asyncio
    async def test_inspect_provider_not_found(self, mock_config, mock_deps):
        """Test inspect_provider raises error for non-existent provider."""
        provider_impl = ProviderImpl(mock_config, mock_deps)

        with pytest.raises(ValueError, match="Provider nonexistent not found"):
            await provider_impl.inspect_provider("nonexistent")

    @pytest.mark.asyncio
    async def test_inspect_provider_malformed_metrics(self, mock_config_malformed_metrics, mock_deps):
        """Test inspect_provider with invalid metrics URL raises validation error."""
        provider_impl = ProviderImpl(mock_config_malformed_metrics, mock_deps)

        with pytest.raises(ValidationError):
            await provider_impl.inspect_provider("test_provider_malformed_metrics")
