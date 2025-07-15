# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.models import ModelType
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import RoutingTable

logger = get_logger(name=__name__, category="core")


async def get_embedding_model_info(
    model_id: str, routing_table: RoutingTable, override_dimension: int | None = None
) -> tuple[str, int]:
    """
    Get embedding model info with auto-dimension lookup.

    This function validates that the specified model is an embedding model
    and returns its embedding dimensions, with support for Matryoshka embeddings
    through dimension overrides.

    Args:
        model_id: The embedding model identifier to look up
        routing_table: Access to the model registry for validation and dimension lookup
        override_dimension: Optional dimension override for Matryoshka models that
                          support variable dimensions (e.g., nomic-embed-text)

    Returns:
        tuple: (model_id, embedding_dimension)

    Raises:
        ValueError: If model not found, not an embedding model, or missing dimension info
    """
    try:
        # Look up the model in the routing table
        model = await routing_table.get_object_by_identifier("model", model_id)  # type: ignore
        if model is None:
            raise ValueError(f"Embedding model '{model_id}' not found in model registry")

        # Validate that this is an embedding model
        if not hasattr(model, "model_type") or model.model_type != ModelType.embedding:
            raise ValueError(
                f"Model '{model_id}' is not an embedding model (type: {getattr(model, 'model_type', 'unknown')})"
            )

        # If override dimension is provided, use it (for Matryoshka embeddings)
        if override_dimension is not None:
            if override_dimension <= 0:
                raise ValueError(f"Override dimension must be positive, got {override_dimension}")
            logger.info(f"Using override dimension {override_dimension} for embedding model '{model_id}'")
            return model_id, override_dimension

        # Extract embedding dimension from model metadata
        if not hasattr(model, "metadata") or not model.metadata:
            raise ValueError(f"Embedding model '{model_id}' has no metadata")

        embedding_dimension = model.metadata.get("embedding_dimension")
        if embedding_dimension is None:
            raise ValueError(f"Embedding model '{model_id}' has no embedding_dimension in metadata")

        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            raise ValueError(f"Invalid embedding_dimension for model '{model_id}': {embedding_dimension}")

        logger.debug(f"Auto-lookup successful for embedding model '{model_id}': dimension {embedding_dimension}")
        return model_id, embedding_dimension

    except Exception as e:
        logger.error(f"Error looking up embedding model info for '{model_id}': {e}")
        raise


async def get_provider_embedding_model_info(
    routing_table: RoutingTable,
    provider_config,
    explicit_model_id: str | None = None,
    explicit_dimension: int | None = None,
) -> tuple[str, int] | None:
    """
    Get embedding model info with provider-level defaults and explicit overrides.

    This function implements the priority order for embedding model selection:
    1. Explicit parameters (from API calls)
    2. Provider config defaults (NEW - from VectorIOConfig)
    3. System default (current fallback behavior)

    Args:
        routing_table: Access to the model registry
        provider_config: The VectorIOConfig object with potential embedding_model defaults
        explicit_model_id: Explicit model ID from API call (highest priority)
        explicit_dimension: Explicit dimension from API call (highest priority)

    Returns:
        tuple: (model_id, embedding_dimension) or None if no model available

    Raises:
        ValueError: If model validation fails
    """
    try:
        # Priority 1: Explicit parameters (existing behavior)
        if explicit_model_id is not None:
            logger.debug(f"Using explicit embedding model: {explicit_model_id}")
            return await get_embedding_model_info(explicit_model_id, routing_table, explicit_dimension)

        # Priority 2: Provider config default (NEW)
        if hasattr(provider_config, "embedding_model") and provider_config.embedding_model:
            logger.info(f"Using provider config default embedding model: {provider_config.embedding_model}")
            override_dim = None
            if hasattr(provider_config, "embedding_dimension") and provider_config.embedding_dimension:
                override_dim = provider_config.embedding_dimension
                logger.info(f"Using provider config dimension override: {override_dim}")

            return await get_embedding_model_info(provider_config.embedding_model, routing_table, override_dim)

        # Priority 3: System default (existing fallback behavior)
        logger.debug("No explicit model or provider default, falling back to system default")
        return await _get_first_embedding_model_fallback(routing_table)

    except Exception as e:
        logger.error(f"Error getting provider embedding model info: {e}")
        raise


async def _get_first_embedding_model_fallback(routing_table: RoutingTable) -> tuple[str, int] | None:
    """
    Fallback to get the first available embedding model (existing behavior).

    This maintains backward compatibility by preserving the original logic
    from VectorIORouter._get_first_embedding_model().
    """
    try:
        # Get all models from the routing table
        all_models = await routing_table.get_all_with_type("model")  # type: ignore

        # Filter for embedding models
        embedding_models = [
            model for model in all_models if hasattr(model, "model_type") and model.model_type == ModelType.embedding
        ]

        if embedding_models:
            dimension = embedding_models[0].metadata.get("embedding_dimension", None)
            if dimension is None:
                raise ValueError(f"Embedding model {embedding_models[0].identifier} has no embedding dimension")

            logger.info(f"System fallback: using first available embedding model {embedding_models[0].identifier}")
            return embedding_models[0].identifier, dimension
        else:
            logger.warning("No embedding models found in the routing table")
            return None
    except Exception as e:
        logger.error(f"Error getting fallback embedding model: {e}")
        return None
