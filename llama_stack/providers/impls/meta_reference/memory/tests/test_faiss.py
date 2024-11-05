import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from llama_stack.providers.impls.meta_reference.memory.faiss import (
    FaissIndex,
    FaissMemoryImpl,
    MEMORY_BANKS_PREFIX,
)
from llama_stack.providers.impls.meta_reference.memory.config import FaissImplConfig
from llama_stack.providers.utils.memory.vector_store import ALL_MINILM_L6_V2_DIMENSION
from llama_stack.apis.memory import (
    Chunk,
    QueryDocumentsResponse,
    VectorMemoryBankDef,
    MemoryBankType,
)


@pytest.fixture
def faiss_index():
    return FaissIndex(dimension=ALL_MINILM_L6_V2_DIMENSION)


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            document_id="doc1",
            content="This is the first test chunk",
            metadata={"test": "metadata1"},
            token_count=7
        ),
        Chunk(
            document_id="doc2",
            content="This is the second test chunk",
            metadata={"test": "metadata2"},
            token_count=7
        ),
    ]


@pytest.fixture
def sample_embeddings():
    return np.array([
        [1.0, 0.0] + [0.0] * (ALL_MINILM_L6_V2_DIMENSION - 2),
        [0.0, 1.0] + [0.0] * (ALL_MINILM_L6_V2_DIMENSION - 2),
    ], dtype=np.float32)


class TestFaissIndex:
    @pytest.mark.asyncio
    async def test_add_chunks(self, faiss_index, sample_chunks, sample_embeddings):
        await faiss_index.add_chunks(sample_chunks, sample_embeddings)
        
        assert len(faiss_index.id_by_index) == 2
        assert len(faiss_index.chunk_by_index) == 2
        assert faiss_index.id_by_index[0] == "doc1"
        assert faiss_index.id_by_index[1] == "doc2"
        assert faiss_index.chunk_by_index[0].content == "This is the first test chunk"
        assert faiss_index.chunk_by_index[1].content == "This is the second test chunk"

    @pytest.mark.asyncio
    async def test_query(self, faiss_index, sample_chunks, sample_embeddings):
        await faiss_index.add_chunks(sample_chunks, sample_embeddings)
        
        # Query vector closer to first chunk
        query_vector = np.array([[0.9, 0.1] + [0.0] * (ALL_MINILM_L6_V2_DIMENSION - 2)], dtype=np.float32)
        response = await faiss_index.query(query_vector, k=2, score_threshold=0.0)
        
        assert isinstance(response, QueryDocumentsResponse)
        assert len(response.chunks) == 2
        assert len(response.scores) == 2
        assert response.chunks[0].document_id == "doc1"
        assert response.chunks[1].document_id == "doc2"

    @pytest.mark.asyncio
    async def test_query_with_threshold(self, faiss_index, sample_chunks, sample_embeddings):
        await faiss_index.add_chunks(sample_chunks, sample_embeddings)
        
        # Query vector far from both chunks
        query_vector = np.array([[0.1, 0.1] + [1.0] * (ALL_MINILM_L6_V2_DIMENSION - 2)], dtype=np.float32)
        response = await faiss_index.query(query_vector, k=2, score_threshold=0.5)
        
        assert isinstance(response, QueryDocumentsResponse)
        assert len(response.chunks) == 0
        assert len(response.scores) == 0


class TestFaissMemoryImpl:
    @pytest.fixture
    def mock_kvstore(self):
        mock = AsyncMock()
        mock.range = AsyncMock(return_value=[])
        return mock

    @pytest.fixture
    def faiss_impl(self, mock_kvstore):
        config = FaissImplConfig()
        impl = FaissMemoryImpl(config)
        impl.kvstore = mock_kvstore
        return impl

    @pytest.mark.asyncio
    async def test_initialize(self, faiss_impl, mock_kvstore):
        # Test empty initialization
        await faiss_impl.initialize()
        mock_kvstore.range.assert_called_once_with(
            MEMORY_BANKS_PREFIX, 
            f"{MEMORY_BANKS_PREFIX}\xff"
        )
        assert len(faiss_impl.cache) == 0

        # Test initialization with existing banks
        bank = VectorMemoryBankDef(
            identifier="test_bank",
            type=MemoryBankType.vector.value,
            embedding_model="all-MiniLM-L6-v2",
            chunk_size_in_tokens=512,
            overlap_size_in_tokens=64,
        )
        mock_kvstore.range.return_value = [bank.json()]
        
        await faiss_impl.initialize()
        assert len(faiss_impl.cache) == 1
        assert "test_bank" in faiss_impl.cache

    @pytest.mark.asyncio
    async def test_register_memory_bank(self, faiss_impl):
        bank = VectorMemoryBankDef(
            identifier="test_bank",
            type=MemoryBankType.vector.value,
            embedding_model="all-MiniLM-L6-v2",
            chunk_size_in_tokens=512,
            overlap_size_in_tokens=64,
        )
        
        await faiss_impl.register_memory_bank(bank)
        
        faiss_impl.kvstore.set.assert_called_once_with(
            key=f"{MEMORY_BANKS_PREFIX}test_bank",
            value=bank.json(),
        )
        assert "test_bank" in faiss_impl.cache
        assert faiss_impl.cache["test_bank"].bank == bank

    @pytest.mark.asyncio
    async def test_register_invalid_bank_type(self, faiss_impl):
        bank = VectorMemoryBankDef(
            identifier="test_bank",
            type="invalid_type",
            embedding_model="all-MiniLM-L6-v2",
            chunk_size_in_tokens=512,
            overlap_size_in_tokens=64,
        )
        
        with pytest.raises(AssertionError):
            await faiss_impl.register_memory_bank(bank)

if __name__ == "__main__":
    pytest.main([__file__])
