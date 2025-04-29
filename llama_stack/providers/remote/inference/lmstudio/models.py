# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.models.models import ModelType
from llama_stack.models.llama.sku_list import CoreModelId
from llama_stack.providers.utils.inference.model_registry import (
    ProviderModelEntry,
)

MODEL_ENTRIES = [
    ProviderModelEntry(
        provider_model_id="meta-llama-3-8b-instruct",
        aliases=[],
        llama_model=CoreModelId.llama3_8b_instruct.value,
        model_type=ModelType.llm,
    ),
    ProviderModelEntry(
        provider_model_id="meta-llama-3-70b-instruct",
        aliases=[],
        llama_model=CoreModelId.llama3_70b_instruct.value,
        model_type=ModelType.llm,
    ),
    ProviderModelEntry(
        provider_model_id="meta-llama-3.1-8b-instruct",
        aliases=[],
        llama_model=CoreModelId.llama3_1_8b_instruct.value,
        model_type=ModelType.llm,
    ),
    ProviderModelEntry(
        provider_model_id="meta-llama-3.1-70b-instruct",
        aliases=[],
        llama_model=CoreModelId.llama3_1_70b_instruct.value,
        model_type=ModelType.llm,
    ),
    ProviderModelEntry(
        provider_model_id="llama-3.2-1b-instruct",
        aliases=[],
        llama_model=CoreModelId.llama3_2_1b_instruct.value,
        model_type=ModelType.llm,
    ),
    ProviderModelEntry(
        provider_model_id="llama-3.2-3b-instruct",
        aliases=[],
        llama_model=CoreModelId.llama3_2_3b_instruct.value,
        model_type=ModelType.llm,
    ),
    ProviderModelEntry(
        provider_model_id="llama-3.3-70b-instruct",
        aliases=[],
        llama_model=CoreModelId.llama3_3_70b_instruct.value,
        model_type=ModelType.llm,
    ),
    # embedding model
    ProviderModelEntry(
        provider_model_id="nomic-embed-text-v1.5",
        model_type=ModelType.embedding,
        metadata={
            "embedding_dimension": 768,
            "context_length": 2048,
        },
    ),
    ProviderModelEntry(
        provider_model_id="all-minilm-l6-v2",
        model_type=ModelType.embedding,
        metadata={
            "embedding_dimension": 384,
        },
    ),
]
