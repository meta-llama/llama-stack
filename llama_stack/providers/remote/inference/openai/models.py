# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.models.models import ModelType
from llama_stack.providers.utils.inference.model_registry import (
    ProviderModelEntry,
)

LLM_MODEL_IDS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/chatgpt-4o-latest",
]


MODEL_ENTRIES = [ProviderModelEntry(provider_model_id=m) for m in LLM_MODEL_IDS] + [
    ProviderModelEntry(
        provider_model_id="openai/text-embedding-3-small",
        model_type=ModelType.embedding,
        metadata={"embedding_dimension": 1536, "context_length": 8192},
    ),
    ProviderModelEntry(
        provider_model_id="openai/text-embedding-3-large",
        model_type=ModelType.embedding,
        metadata={"embedding_dimension": 3072, "context_length": 8192},
    ),
]
