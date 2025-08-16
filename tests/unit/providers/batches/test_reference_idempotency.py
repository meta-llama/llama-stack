# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Tests for idempotency functionality in the reference batches provider.

This module tests the optional idempotency feature that allows clients to provide
an idempotency token (idem_tok) to ensure that repeated requests with the same token
and parameters return the same batch, while requests with the same token but different
parameters result in a conflict error.

Test Categories:
1. Core Idempotency: Same parameters with same token return same batch
2. Parameter Independence: Different parameters without tokens create different batches
3. Conflict Detection: Same token with different parameters raises ConflictError

Tests by Category:

1. Core Idempotency:
   - test_idempotent_batch_creation_same_params
   - test_idempotent_batch_creation_metadata_order_independence

2. Parameter Independence:
   - test_non_idempotent_behavior_without_token
   - test_different_idempotency_tokens_create_different_batches

3. Conflict Detection:
   - test_same_idem_tok_different_params_conflict (parametrized: input_file_id, metadata values, metadata None vs {})

Key Behaviors Tested:
- Idempotent batch creation when idem_tok provided with identical parameters
- Metadata order independence for consistent batch ID generation
- Non-idempotent behavior when no idem_tok provided (random UUIDs)
- Conflict detection for parameter mismatches with same idempotency token
- Deterministic ID generation based solely on idempotency token
- Proper error handling with detailed conflict messages including token and error codes
- Protection against idempotency token reuse with different request parameters
"""

import asyncio

import pytest

from llama_stack.apis.common.errors import ConflictError


class TestReferenceBatchesIdempotency:
    """Test suite for idempotency functionality in the reference implementation."""

    async def test_idempotent_batch_creation_same_params(self, provider, sample_batch_data):
        """Test that creating batches with identical parameters returns the same batch when idem_tok is provided."""

        del sample_batch_data["metadata"]

        batch1 = await provider.create_batch(
            **sample_batch_data,
            metadata={"test": "value1", "other": "value2"},
            idem_tok="unique-token-1",
        )

        # sleep for 1 second to allow created_at timestamps to be different
        await asyncio.sleep(1)

        batch2 = await provider.create_batch(
            **sample_batch_data,
            metadata={"other": "value2", "test": "value1"},  # Different order
            idem_tok="unique-token-1",
        )

        assert batch1.id == batch2.id
        assert batch1.input_file_id == batch2.input_file_id
        assert batch1.metadata == batch2.metadata
        assert batch1.created_at == batch2.created_at

    async def test_different_idempotency_tokens_create_different_batches(self, provider, sample_batch_data):
        """Test that different idempotency tokens create different batches even with same params."""
        batch1 = await provider.create_batch(
            **sample_batch_data,
            idem_tok="token-A",
        )

        batch2 = await provider.create_batch(
            **sample_batch_data,
            idem_tok="token-B",
        )

        assert batch1.id != batch2.id

    @pytest.mark.parametrize(
        "param_name,first_value,second_value",
        [
            ("input_file_id", "file_001", "file_002"),
            ("metadata", {"test": "value1"}, {"test": "value2"}),
            ("metadata", None, {}),
        ],
    )
    async def test_same_idem_tok_different_params_conflict(
        self, provider, sample_batch_data, param_name, first_value, second_value
    ):
        """Test that same idem_tok with different parameters raises conflict error."""
        sample_batch_data["idem_tok"] = "same-token"

        sample_batch_data[param_name] = first_value

        batch1 = await provider.create_batch(**sample_batch_data)

        with pytest.raises(ConflictError, match="Idempotency token.*was previously used with different parameters"):
            sample_batch_data[param_name] = second_value
            await provider.create_batch(**sample_batch_data)

        retrieved_batch = await provider.retrieve_batch(batch1.id)
        assert retrieved_batch.id == batch1.id
        assert getattr(retrieved_batch, param_name) == first_value
