# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Patch for the provider impl to fix the health check for the FAISS provider.
It is the workaround fix with current implementation if place for get_providers_health
as it returns a dict mapping API names to a single health response, but list_providers
expects a dict mapping API names to a dict of provider IDs to health responses.
"""

import logging

import faiss

from llama_stack.distribution.providers import ProviderImpl
from llama_stack.providers.datatypes import HealthResponse, HealthStatus

logger = logging.getLogger(__name__)

# Store the original methods
original_list_providers = ProviderImpl.list_providers

VECTOR_DIMENSION = 128  # sample dimension


# Helper method to check FAISS health directly
async def check_faiss_health():
    """Check the health of the FAISS vector database directly."""
    try:
        # Create FAISS index to verify readiness
        faiss.IndexFlatL2(VECTOR_DIMENSION)
        return HealthResponse(status=HealthStatus.OK)
    except Exception as e:
        return HealthResponse(status=HealthStatus.ERROR, message=f"FAISS health check failed: {str(e)}")


async def patched_list_providers(self):
    """Patched version of list_providers to include FAISS health check."""
    logger.debug("Using patched list_providers method")
    # Get the original response
    response = await original_list_providers(self)
    # To find the FAISS provider in the response
    for provider in response.data:
        if provider.provider_id == "faiss" and provider.api == "vector_io":
            health_result = await check_faiss_health()
            logger.debug("FAISS health check result: %s", health_result)
            provider.health = health_result
            logger.debug("Updated FAISS health to: %s", provider.health)
    return response


# Apply the patch by replacing the original method with patched version
# Added type: ignore because mypy cannot infer the correct type
# The typing error doesn't affect runtime behavior - it's only a static type check warning
ProviderImpl.list_providers = patched_list_providers  # type: ignore
logger.debug("Successfully applied patch for FAISS provider health check")
