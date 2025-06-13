import os
import pytest
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

from llama_stack.apis.inference import Message
from llama_stack.apis.synthetic_data_generation import (
    SyntheticDataGeneration,
    SyntheticDataGenerationResponse,
    FilteringFunction,
)
from llama_stack.providers.inline.synthetic_data_generation.synthetic_data_kit.config import (
    SyntheticDataKitConfig,
)
from llama_stack.providers.inline.synthetic_data_generation.synthetic_data_kit.synthetic_data_kit import (
    SyntheticDataKitProvider,
)


def test_config_defaults():
    """Test default configuration values"""
    config = SyntheticDataKitConfig()
    assert config.llm["provider"] == "vllm"
    assert config.llm["model"] == "meta-llama/Llama-3.2-3B-Instruct"
    assert config.vllm["api_base"] == "http://localhost:8000/v1"
    assert config.generation["temperature"] == 0.7
    assert config.generation["chunk_size"] == 4000
    assert config.curate["threshold"] == 7.0


def test_sample_run_config():
    """Test sample configuration with environment variables"""
    # Test default configuration
    config = SyntheticDataKitConfig.sample_run_config()
    assert isinstance(config, SyntheticDataKitConfig)
    assert config.llm["model"] == "meta-llama/Llama-3.2-3B-Instruct"
    
    # Test environment variable override
    os.environ["SYNTHETIC_DATA_KIT_MODEL"] = "meta-llama/Llama-3.2-7B-Instruct"
    config = SyntheticDataKitConfig.sample_run_config()
    assert config.llm["model"] == "meta-llama/Llama-3.2-7B-Instruct"


@pytest.fixture
def mock_sdk():
    """Create a mock SDK instance"""
    with patch("synthetic_data_kit.SyntheticDataKit") as mock:
        sdk_instance = MagicMock()
        sdk_instance.create = AsyncMock()
        sdk_instance.curate = AsyncMock()
        mock.return_value = sdk_instance
        yield sdk_instance


@pytest.fixture
def config():
    return SyntheticDataKitConfig()


@pytest.fixture
def provider(config: SyntheticDataKitConfig, mock_sdk):
    return SyntheticDataKitProvider(config)


@pytest.mark.asyncio
async def test_synthetic_data_generate_basic(provider: SyntheticDataGeneration, mock_sdk):
    # Setup mock response
    mock_sdk.create.return_value = {
        "synthetic_data": [{"question": "What is ML?", "answer": "Machine learning..."}],
        "statistics": {"count": 1}
    }
    
    dialogs = [Message(role="user", content="What is machine learning?")]
    response = await provider.synthetic_data_generate(
        dialogs=dialogs,
        filtering_function=FilteringFunction.none,
    )
    
    # Verify SDK was called correctly
    mock_sdk.create.assert_called_once_with("What is machine learning?", type="qa")
    assert isinstance(response, SyntheticDataGenerationResponse)
    assert len(response.synthetic_data) == 1
    assert response.statistics == {"count": 1}


@pytest.mark.asyncio
async def test_synthetic_data_generate_with_filtering(provider: SyntheticDataGeneration, mock_sdk):
    # Setup mock responses
    mock_sdk.create.return_value = {
        "synthetic_data": [{"question": "What is quantum?", "answer": "Quantum..."}],
    }
    mock_sdk.curate.return_value = {
        "synthetic_data": [{"question": "What is quantum?", "answer": "Quantum..."}],
        "statistics": {"threshold": 7.5}
    }
    
    dialogs = [Message(role="user", content="Explain quantum computing.")]
    response = await provider.synthetic_data_generate(
        dialogs=dialogs,
        filtering_function=FilteringFunction.top_k,
    )
    
    # Verify both create and curate were called
    mock_sdk.create.assert_called_once_with("Explain quantum computing.", type="qa")
    mock_sdk.curate.assert_called_once()
    assert isinstance(response, SyntheticDataGenerationResponse)
    assert response.statistics["threshold"] == 7.5


@pytest.mark.asyncio
async def test_synthetic_data_generate_multiple_messages(provider: SyntheticDataGeneration, mock_sdk):
    mock_sdk.create.return_value = {
        "synthetic_data": [{"question": "What is deep learning?", "answer": "Deep..."}],
        "statistics": {"count": 1}
    }
    
    dialogs = [
        Message(role="user", content="What is deep learning?"),
        Message(role="assistant", content="Deep learning is..."),
        Message(role="user", content="Can you explain more?")
    ]
    
    response = await provider.synthetic_data_generate(
        dialogs=dialogs,
        filtering_function=FilteringFunction.none,
    )
    
    # Verify content was joined correctly
    expected_content = "What is deep learning?\nDeep learning is...\nCan you explain more?"
    mock_sdk.create.assert_called_once_with(expected_content, type="qa")
    assert isinstance(response, SyntheticDataGenerationResponse)
    assert response.synthetic_data is not None 