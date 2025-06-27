import os
import pytest
from typing import cast

from llama_stack.apis.inference import Message
from llama_stack.apis.synthetic_data_generation import (
    SyntheticDataGeneration,
    FilteringFunction,
)
from llama_stack.apis.synthetic_data_generation.providers import get_provider_impl
from llama_stack.distribution.client import LlamaStackAsLibraryClient


@pytest.fixture
async def client():
    # Use LlamaStackAsLibraryClient for inline testing
    return LlamaStackAsLibraryClient()


@pytest.mark.asyncio
async def test_synthetic_data_kit_provider_integration(client: LlamaStackAsLibraryClient):
    provider = await get_provider_impl()
    assert isinstance(provider, SyntheticDataGeneration)

    # Test single message generation
    dialogs = [
        Message(role="user", content="What is artificial intelligence?"),
    ]
    
    response = await provider.synthetic_data_generate(
        dialogs=dialogs,
        filtering_function=FilteringFunction.none,
    )
    
    assert response.synthetic_data is not None
    assert len(response.synthetic_data) > 0
    assert all(isinstance(item, dict) for item in response.synthetic_data)
    assert all("question" in item and "answer" in item for item in response.synthetic_data)


@pytest.mark.asyncio
async def test_synthetic_data_kit_provider_with_filtering(client: LlamaStackAsLibraryClient):
    provider = await get_provider_impl()
    
    # Test generation with filtering
    dialogs = [
        Message(role="user", content="Explain quantum computing."),
        Message(role="assistant", content="Quantum computing uses quantum mechanics..."),
    ]
    
    response = await provider.synthetic_data_generate(
        dialogs=dialogs,
        filtering_function=FilteringFunction.top_k,
    )
    
    assert response.synthetic_data is not None
    assert len(response.synthetic_data) > 0
    assert response.statistics is not None
    assert "threshold" in response.statistics


@pytest.mark.asyncio
async def test_synthetic_data_kit_provider_error_handling(client: LlamaStackAsLibraryClient):
    provider = await get_provider_impl()
    
    # Test with empty dialogs
    with pytest.raises(ValueError):
        await provider.synthetic_data_generate(
            dialogs=[],
            filtering_function=FilteringFunction.none,
        )
    
    # Test with invalid model
    with pytest.raises(RuntimeError):
        await provider.synthetic_data_generate(
            dialogs=[Message(role="user", content="Test")],
            filtering_function=FilteringFunction.none,
            model="invalid-model",
        )


@pytest.mark.asyncio
async def test_synthetic_data_kit_provider_with_env_config(client: LlamaStackAsLibraryClient):
    # Set environment variables for testing
    os.environ["SYNTHETIC_DATA_KIT_MODEL"] = "meta-llama/Llama-3.2-7B-Instruct"
    
    provider = await get_provider_impl()
    dialogs = [
        Message(role="user", content="What is deep learning?"),
        Message(role="assistant", content="Deep learning is a subset of machine learning..."),
    ]
    
    response = await provider.synthetic_data_generate(
        dialogs=dialogs,
        filtering_function=FilteringFunction.none,
    )
    
    assert response.synthetic_data is not None
    assert len(response.synthetic_data) > 0
    # Clean up environment
    del os.environ["SYNTHETIC_DATA_KIT_MODEL"] 