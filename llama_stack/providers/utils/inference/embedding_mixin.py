# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from llama_stack.apis.inference import (
    EmbeddingsResponse,
    EmbeddingTaskType,
    InterleavedContentItem,
    ModelStore,
    TextTruncation,
)
from llama_stack.providers.utils.inference.prompt_adapter import interleaved_content_as_str

EMBEDDING_MODELS = {}


log = logging.getLogger(__name__)


class SentenceTransformerEmbeddingMixin:
    model_store: ModelStore

    async def embeddings(
        self,
        model_id: str,
        contents: List[str] | List[InterleavedContentItem],
        text_truncation: Optional[TextTruncation] = TextTruncation.none,
        output_dimension: Optional[int] = None,
        task_type: Optional[EmbeddingTaskType] = None,
    ) -> EmbeddingsResponse:
        model = await self.model_store.get_model(model_id)
        embedding_model = self._load_sentence_transformer_model(model.provider_resource_id)
        embeddings = embedding_model.encode(
            [interleaved_content_as_str(content) for content in contents], show_progress_bar=False
        )
        return EmbeddingsResponse(embeddings=embeddings)

    def _load_sentence_transformer_model(self, model: str) -> "SentenceTransformer":
        global EMBEDDING_MODELS

        loaded_model = EMBEDDING_MODELS.get(model)
        if loaded_model is not None:
            return loaded_model

        log.info(f"Loading sentence transformer for {model}...")
        from sentence_transformers import SentenceTransformer

        loaded_model = SentenceTransformer(model)
        EMBEDDING_MODELS[model] = loaded_model
        return loaded_model
