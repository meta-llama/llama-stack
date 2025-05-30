"""
Unit tests for the FAISS provider health check implementation via provider patch.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from llama_stack.distribution.providers import ProviderImpl
from llama_stack.providers.datatypes import HealthResponse, HealthStatus
from llama_stack.providers.inline.vector_io.faiss.provider_patch import (
    check_faiss_health,
    patched_list_providers,
)


class TestFaissProviderPatch(unittest.TestCase):
    """Test cases for the FAISS provider patch."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider_impl = MagicMock(spec=ProviderImpl)
        self.mock_response = MagicMock()
        self.mock_response.data = []
        # Set up the original list_providers method
        self.original_list_providers = AsyncMock(return_value=self.mock_response)

    async def test_check_faiss_health_success(self):
        """Test the check_faiss_health function when FAISS is working properly."""
        with patch("faiss.IndexFlatL2") as mock_index:
            mock_index.return_value = MagicMock()
            # Call the health check function
            result = await check_faiss_health()

            self.assertEqual(result.status, HealthStatus.OK)
            mock_index.assert_called_once()

    async def test_check_faiss_health_failure(self):
        """Test the check_faiss_health function when FAISS fails."""
        with patch("faiss.IndexFlatL2") as mock_index:
            # Configure the mock to simulate a failure
            mock_index.side_effect = Exception("FAISS initialization failed")
            result = await check_faiss_health()

            self.assertEqual(result.status, HealthStatus.ERROR)
            self.assertIn("FAISS health check failed", result.message)
            mock_index.assert_called_once()

    async def test_patched_list_providers_no_faiss(self):
        """Test the patched_list_providers method when no FAISS provider is found."""
        # Set up the mock response with NO FAISS provider
        self.mock_response.data = [
            MagicMock(provider_id="other", api="vector_io"),
            MagicMock(provider_id="faiss", api="other_api"),
        ]
        with patch(
            "llama_stack.providers.inline.vector_io.faiss.provider_patch.original_list_providers",
            self.original_list_providers
        ):
            result = await patched_list_providers(self.provider_impl)

            self.assertEqual(result, self.mock_response)
            self.original_list_providers.assert_called_once_with(self.provider_impl)
            # Verify that no health checks were performed
            for provider in result.data:
                self.assertNotEqual(provider.provider_id, "faiss")

    async def test_patched_list_providers_with_faiss(self):
        """Test the patched_list_providers method when a FAISS provider is found."""
        # Create a mock FAISS provider
        mock_faiss_provider = MagicMock(provider_id="faiss", api="vector_io")
        mock_faiss_provider.health = MagicMock(
            return_value=HealthResponse(status=HealthStatus.NOT_IMPLEMENTED)
        )
        # Set up the mock response with a FAISS provider
        self.mock_response.data = [
            MagicMock(provider_id="other", api="vector_io"),
            mock_faiss_provider,
        ]
        with patch(
            "llama_stack.providers.inline.vector_io.faiss.provider_patch.original_list_providers",
            self.original_list_providers
        ), \
            patch(
                "llama_stack.providers.inline.vector_io.faiss.provider_patch.check_faiss_health"
            ) as mock_health:
            mock_health.return_value = HealthResponse(status=HealthStatus.OK)
            result = await patched_list_providers(self.provider_impl)
            self.assertEqual(result, self.mock_response)
            self.original_list_providers.assert_called_once_with(self.provider_impl)
            mock_health.assert_called_once()
            # Verify that the FAISS provider's health was updated
            for provider in result.data:
                if provider.provider_id == "faiss" and provider.api == "vector_io":
                    self.assertEqual(provider.health.status, HealthStatus.OK)

    async def test_patched_list_providers_with_faiss_health_failure(self):
        """Test the patched_list_providers method when the FAISS health check fails."""
        mock_faiss_provider = MagicMock(provider_id="faiss", api="vector_io")
        mock_faiss_provider.health = MagicMock(
            return_value=HealthResponse(status=HealthStatus.NOT_IMPLEMENTED)
        )
        self.mock_response.data = [
            MagicMock(provider_id="other", api="vector_io"),
            mock_faiss_provider,
        ]
        with patch(
            "llama_stack.providers.inline.vector_io.faiss.provider_patch.original_list_providers",
            self.original_list_providers), \
            patch(
                "llama_stack.providers.inline.vector_io.faiss.provider_patch.check_faiss_health"
            ) as mock_health:
            # Configure the mock health check to simulate a failure
            error_response = HealthResponse(
                status=HealthStatus.ERROR,
                message="FAISS health check failed: Test error"
            )
            mock_health.return_value = error_response

            result = await patched_list_providers(self.provider_impl)
            self.assertEqual(result, self.mock_response)
            self.original_list_providers.assert_called_once_with(self.provider_impl)
            mock_health.assert_called_once()
            # Verify that the FAISS provider's health was updated with the error
            for provider in result.data:
                if provider.provider_id == "faiss" and provider.api == "vector_io":
                    self.assertEqual(provider.health.status, HealthStatus.ERROR)
                    self.assertEqual(
                        provider.health.message, "FAISS health check failed: Test error"
                    )

    async def test_patched_list_providers_with_exception(self):
        """Test the patched_list_providers method when an exception occurs during health check."""
        mock_faiss_provider = MagicMock(provider_id="faiss", api="vector_io")
        mock_faiss_provider.health = MagicMock(
            return_value=HealthResponse(status=HealthStatus.NOT_IMPLEMENTED)
        )

        self.mock_response.data = [
            MagicMock(provider_id="other", api="vector_io"),
            mock_faiss_provider,
        ]
        with patch(
            "llama_stack.providers.inline.vector_io.faiss.provider_patch.original_list_providers",
            self.original_list_providers
        ), \
        patch(
            "llama_stack.providers.inline.vector_io.faiss.provider_patch.check_faiss_health"
        ) as mock_health:
            # Configure the mock health check to raise an exception
            mock_health.side_effect = Exception("Unexpected error")
            result = await patched_list_providers(self.provider_impl)

            self.assertEqual(result, self.mock_response)
            self.original_list_providers.assert_called_once_with(self.provider_impl)
            mock_health.assert_called_once()
            # Verify that the FAISS provider's health was updated with an error
            for provider in result.data:
                if provider.provider_id == "faiss" and provider.api == "vector_io":
                    self.assertEqual(provider.health.status, HealthStatus.ERROR)
                    self.assertIn("Failed to check FAISS health", provider.health.message)


if __name__ == "__main__":
    unittest.main()
