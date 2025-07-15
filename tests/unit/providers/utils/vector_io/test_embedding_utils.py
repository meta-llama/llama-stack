# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

import pytest

from llama_stack.apis.models import ModelType
from llama_stack.providers.utils.vector_io.embedding_utils import (
    _get_first_embedding_model_fallback,
    get_embedding_model_info,
    get_provider_embedding_model_info,
)


class MockModel:
    """Mock model object for testing."""

    def __init__(self, identifier: str, model_type: ModelType, metadata: dict | None = None):
        self.identifier = identifier
        self.model_type = model_type
        self.metadata = metadata


class MockConfig:
    """Mock provider config for testing."""

    def __init__(self, embedding_model: str | None = None, embedding_dimension: int | None = None):
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension


@pytest.fixture
def mock_routing_table():
    """Create a mock routing table for testing."""
    routing_table = AsyncMock()

    # Mock embedding models
    embedding_models = [
        MockModel(identifier="all-MiniLM-L6-v2", model_type=ModelType.embedding, metadata={"embedding_dimension": 384}),
        MockModel(identifier="nomic-embed-text", model_type=ModelType.embedding, metadata={"embedding_dimension": 768}),
    ]

    # Mock LLM model (should be filtered out)
    llm_model = MockModel(identifier="llama-3.1-8b", model_type=ModelType.llm, metadata={})

    all_models = embedding_models + [llm_model]

    async def mock_get_object_by_identifier(type_name: str, identifier: str):
        if type_name == "model":
            for model in all_models:
                if model.identifier == identifier:
                    return model
        return None

    async def mock_get_all_with_type(type_name: str):
        if type_name == "model":
            return all_models
        return []

    routing_table.get_object_by_identifier.side_effect = mock_get_object_by_identifier
    routing_table.get_all_with_type.side_effect = mock_get_all_with_type

    return routing_table


class TestGetEmbeddingModelInfo:
    """Test the core get_embedding_model_info function."""

    @pytest.mark.asyncio
    async def test_valid_embedding_model(self, mock_routing_table):
        """Test successful lookup of a valid embedding model."""
        model_id, dimension = await get_embedding_model_info("all-MiniLM-L6-v2", mock_routing_table)

        assert model_id == "all-MiniLM-L6-v2"
        assert dimension == 384

    @pytest.mark.asyncio
    async def test_embedding_model_with_override_dimension(self, mock_routing_table):
        """Test Matryoshka embedding with dimension override."""
        model_id, dimension = await get_embedding_model_info(
            "nomic-embed-text", mock_routing_table, override_dimension=256
        )

        assert model_id == "nomic-embed-text"
        assert dimension == 256  # Should use override, not default 768

    @pytest.mark.asyncio
    async def test_model_not_found(self, mock_routing_table):
        """Test error when model doesn't exist."""
        with pytest.raises(ValueError, match="not found in model registry"):
            await get_embedding_model_info("non-existent-model", mock_routing_table)

    @pytest.mark.asyncio
    async def test_non_embedding_model(self, mock_routing_table):
        """Test error when model is not an embedding model."""
        with pytest.raises(ValueError, match="is not an embedding model"):
            await get_embedding_model_info("llama-3.1-8b", mock_routing_table)

    @pytest.mark.asyncio
    async def test_model_missing_dimension_metadata(self, mock_routing_table):
        """Test error when embedding model has no dimension metadata."""
        # Create a model with non-empty metadata dict missing embedding_dimension
        bad_model = MockModel(
            identifier="bad-embedding-model",
            model_type=ModelType.embedding,
            metadata={"some_other_field": "value"},  # Non-empty but missing embedding_dimension
        )

        async def mock_get_bad_model(type_name: str, identifier: str):
            if type_name == "model" and identifier == "bad-embedding-model":
                return bad_model
            return await mock_routing_table.get_object_by_identifier(type_name, identifier)

        mock_routing_table.get_object_by_identifier.side_effect = mock_get_bad_model

        with pytest.raises(ValueError, match="has no embedding_dimension in metadata"):
            await get_embedding_model_info("bad-embedding-model", mock_routing_table)

    @pytest.mark.asyncio
    async def test_invalid_override_dimension(self, mock_routing_table):
        """Test error with invalid override dimension."""
        with pytest.raises(ValueError, match="Override dimension must be positive"):
            await get_embedding_model_info("all-MiniLM-L6-v2", mock_routing_table, override_dimension=0)

        with pytest.raises(ValueError, match="Override dimension must be positive"):
            await get_embedding_model_info("all-MiniLM-L6-v2", mock_routing_table, override_dimension=-10)


class TestGetProviderEmbeddingModelInfo:
    """Test the provider-level embedding model selection with priority system."""

    @pytest.mark.asyncio
    async def test_priority_1_explicit_parameters(self, mock_routing_table):
        """Test highest priority: explicit parameters."""
        config = MockConfig(embedding_model="nomic-embed-text", embedding_dimension=512)

        # Explicit parameters should override config
        result = await get_provider_embedding_model_info(
            routing_table=mock_routing_table,
            provider_config=config,
            explicit_model_id="all-MiniLM-L6-v2",  # Should use this
            explicit_dimension=256,  # Should use this
        )

        assert result is not None
        model_id, dimension = result
        assert model_id == "all-MiniLM-L6-v2"
        assert dimension == 256

    @pytest.mark.asyncio
    async def test_priority_2_provider_config_defaults(self, mock_routing_table):
        """Test middle priority: provider config defaults."""
        config = MockConfig(embedding_model="nomic-embed-text", embedding_dimension=512)

        # No explicit parameters, should use config
        model_id, dimension = await get_provider_embedding_model_info(
            routing_table=mock_routing_table, provider_config=config, explicit_model_id=None, explicit_dimension=None
        )

        assert model_id == "nomic-embed-text"
        assert dimension == 512  # Config override

    @pytest.mark.asyncio
    async def test_priority_2_provider_config_model_only(self, mock_routing_table):
        """Test provider config with model but no dimension override."""
        config = MockConfig(embedding_model="all-MiniLM-L6-v2")  # No dimension override

        model_id, dimension = await get_provider_embedding_model_info(
            routing_table=mock_routing_table, provider_config=config, explicit_model_id=None, explicit_dimension=None
        )

        assert model_id == "all-MiniLM-L6-v2"
        assert dimension == 384  # Auto-lookup from model metadata

    @pytest.mark.asyncio
    async def test_priority_3_system_default(self, mock_routing_table):
        """Test lowest priority: system default fallback."""
        config = MockConfig()  # No defaults set

        model_id, dimension = await get_provider_embedding_model_info(
            routing_table=mock_routing_table, provider_config=config, explicit_model_id=None, explicit_dimension=None
        )

        # Should get first available embedding model
        assert model_id == "all-MiniLM-L6-v2"
        assert dimension == 384

    @pytest.mark.asyncio
    async def test_no_provider_config(self, mock_routing_table):
        """Test with None provider config."""
        model_id, dimension = await get_provider_embedding_model_info(
            routing_table=mock_routing_table, provider_config=None, explicit_model_id=None, explicit_dimension=None
        )

        # Should fall back to system default
        assert model_id == "all-MiniLM-L6-v2"
        assert dimension == 384

    @pytest.mark.asyncio
    async def test_no_embedding_models_available(self, mock_routing_table):
        """Test when no embedding models are available."""

        # Mock routing table with no embedding models
        async def mock_get_all_empty(type_name: str):
            return []  # No models

        mock_routing_table.get_all_with_type.side_effect = mock_get_all_empty

        config = MockConfig()

        result = await get_provider_embedding_model_info(
            routing_table=mock_routing_table, provider_config=config, explicit_model_id=None, explicit_dimension=None
        )

        assert result is None


class TestGetFirstEmbeddingModelFallback:
    """Test the fallback function for system defaults."""

    @pytest.mark.asyncio
    async def test_successful_fallback(self, mock_routing_table):
        """Test successful fallback to first embedding model."""
        model_id, dimension = await _get_first_embedding_model_fallback(mock_routing_table)

        assert model_id == "all-MiniLM-L6-v2"
        assert dimension == 384

    @pytest.mark.asyncio
    async def test_no_embedding_models_fallback(self, mock_routing_table):
        """Test fallback when no embedding models exist."""

        # Mock empty model list
        async def mock_get_all_empty(type_name: str):
            return []

        mock_routing_table.get_all_with_type.side_effect = mock_get_all_empty

        result = await _get_first_embedding_model_fallback(mock_routing_table)
        assert result is None

    @pytest.mark.asyncio
    async def test_embedding_model_missing_dimension_fallback(self, mock_routing_table):
        """Test fallback when embedding model has no dimension - should return None."""
        bad_model = MockModel(
            identifier="bad-embedding",
            model_type=ModelType.embedding,
            metadata={},  # Missing dimension
        )

        async def mock_get_all_bad(type_name: str):
            return [bad_model] if type_name == "model" else []

        mock_routing_table.get_all_with_type.side_effect = mock_get_all_bad

        # The function should return None (not raise) when model has no dimension
        result = await _get_first_embedding_model_fallback(mock_routing_table)
        assert result is None


class TestBackwardCompatibility:
    """Test that the new system maintains backward compatibility."""

    @pytest.mark.asyncio
    async def test_explicit_model_still_works(self, mock_routing_table):
        """Test that explicitly specifying embedding model still works as before."""
        model_id, dimension = await get_embedding_model_info("all-MiniLM-L6-v2", mock_routing_table)

        assert model_id == "all-MiniLM-L6-v2"
        assert dimension == 384

    @pytest.mark.asyncio
    async def test_system_fallback_unchanged(self, mock_routing_table):
        """Test that system fallback behavior is unchanged."""
        # This should behave exactly like the old _get_first_embedding_model
        model_id, dimension = await _get_first_embedding_model_fallback(mock_routing_table)

        assert model_id == "all-MiniLM-L6-v2"
        assert dimension == 384


class TestMatryoshkaEmbeddings:
    """Test specific Matryoshka embedding scenarios."""

    @pytest.mark.asyncio
    async def test_nomic_embed_text_default(self, mock_routing_table):
        """Test nomic-embed-text with default dimension."""
        model_id, dimension = await get_embedding_model_info("nomic-embed-text", mock_routing_table)

        assert model_id == "nomic-embed-text"
        assert dimension == 768  # Default dimension

    @pytest.mark.asyncio
    async def test_nomic_embed_text_override(self, mock_routing_table):
        """Test nomic-embed-text with dimension override."""
        model_id, dimension = await get_embedding_model_info(
            "nomic-embed-text", mock_routing_table, override_dimension=256
        )

        assert model_id == "nomic-embed-text"
        assert dimension == 256  # Overridden dimension

    @pytest.mark.asyncio
    async def test_provider_config_matryoshka_override(self, mock_routing_table):
        """Test provider config with Matryoshka dimension override."""
        config = MockConfig(
            embedding_model="nomic-embed-text",
            embedding_dimension=128,  # Custom dimension
        )

        model_id, dimension = await get_provider_embedding_model_info(
            routing_table=mock_routing_table, provider_config=config, explicit_model_id=None, explicit_dimension=None
        )

        assert model_id == "nomic-embed-text"
        assert dimension == 128  # Should use provider config override
