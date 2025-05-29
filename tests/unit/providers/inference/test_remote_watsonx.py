from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

from llama_stack.providers.datatypes import HealthStatus
from llama_stack.providers.remote.inference.watsonx.config import WatsonXConfig
from llama_stack.providers.remote.inference.watsonx.watsonx import WatsonXInferenceAdapter

@pytest.fixture
def watsonx_config():
    """Create a WatsonXConfig fixture for testing."""
    return WatsonXConfig(
        url="https://test-watsonx-url.ibm.com",
        api_key="test-api-key",
        project_id="test-project-id",
        model_id="test-model-id"
    )

@pytest_asyncio.fixture
async def watsonx_inference_adapter(watsonx_config):
    """Create a WatsonX InferenceAdapter fixture for testing."""
    adapter = WatsonXInferenceAdapter(watsonx_config)
    await adapter.initialize()
    return adapter

@pytest.mark.asyncio
async def test_health_success(watsonx_inference_adapter):
    """
    Test the health status of WatsonX InferenceAdapter when the connection is successful.
    This test verifies that the health method returns a HealthResponse with status OK, only
    when the connection to the WatsonX server is successful.
    """
    # Mock the _get_client method to return a mock model
    mock_model = MagicMock()
    mock_model.generate.return_value = "test response"

    with patch.object(watsonx_inference_adapter, '_get_client', return_value=mock_model):
        health_response = await watsonx_inference_adapter.health()
        # Verify the response
        assert health_response["status"] == HealthStatus.OK
        mock_model.generate.assert_called_once_with("test")

@pytest.mark.asyncio
async def test_health_failure(watsonx_inference_adapter):
    """
    Test the health method of WatsonX InferenceAdapter when the connection fails.
    This test verifies that the health method returns a HealthResponse with status ERROR,
    with the exception error message.
    """
    mock_model = MagicMock()
    mock_model.generate.side_effect = Exception("Connection failed")
    with patch.object(watsonx_inference_adapter, '_get_client', return_value=mock_model):
        health_response = await watsonx_inference_adapter.health()
        assert health_response["status"] == HealthStatus.ERROR
        assert "Health check failed: Connection failed" in health_response["message"]
