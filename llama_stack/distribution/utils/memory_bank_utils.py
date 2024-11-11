# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.apis.memory_banks.memory_banks import (
    BankParams,
    GraphMemoryBank,
    KeyValueMemoryBank,
    KeywordMemoryBank,
    MemoryBank,
    MemoryBankType,
    VectorMemoryBank,
    VectorMemoryBankParams,
)


def build_memory_bank(
    memory_bank_id: str,
    params: BankParams,
    provider_id: str,
    provider_memorybank_id: str,
) -> MemoryBank:
    if params.memory_bank_type == MemoryBankType.vector.value:
        assert isinstance(params, VectorMemoryBankParams)
        memory_bank = VectorMemoryBank(
            identifier=memory_bank_id,
            provider_id=provider_id,
            provider_resource_id=provider_memorybank_id,
            memory_bank_type=params.memory_bank_type,
            embedding_model=params.embedding_model,
            chunk_size_in_tokens=params.chunk_size_in_tokens,
            overlap_size_in_tokens=params.overlap_size_in_tokens,
        )
    elif params.memory_bank_type == MemoryBankType.keyvalue.value:
        memory_bank = KeyValueMemoryBank(
            identifier=memory_bank_id,
            provider_id=provider_id,
            provider_resource_id=provider_memorybank_id,
            memory_bank_type=params.memory_bank_type,
        )
    elif params.memory_bank_type == MemoryBankType.keyword.value:
        memory_bank = KeywordMemoryBank(
            identifier=memory_bank_id,
            provider_id=provider_id,
            provider_resource_id=provider_memorybank_id,
            memory_bank_type=params.memory_bank_type,
        )
    elif params.memory_bank_type == MemoryBankType.graph.value:
        memory_bank = GraphMemoryBank(
            identifier=memory_bank_id,
            provider_id=provider_id,
            provider_resource_id=provider_memorybank_id,
            memory_bank_type=params.memory_bank_type,
        )
    else:
        raise ValueError(f"Unknown memory bank type: {params.memory_bank_type}")
    return memory_bank
