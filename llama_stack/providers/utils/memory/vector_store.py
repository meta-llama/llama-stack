# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


from llama_stack.apis.common.content_types import (
    InterleavedContent,
    TextContentItem,
)
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import Chunk, QueryChunksResponse
from llama_stack.providers.datatypes import Api
from llama_stack.providers.utils.inference.prompt_adapter import (
    interleaved_content_as_str,
)
from numpy.typing import NDArray


log = logging.getLogger(__name__)


def parse_data_url(data_url: str):
    data_url_pattern = re.compile(
        r"^"
        r"data:"
        r"(?P<mimetype>[\w/\-+.]+)"
        r"(?P<charset>;charset=(?P<encoding>[\w-]+))?"
        r"(?P<base64>;base64)?"
        r",(?P<data>.*)"
        r"$",
        re.DOTALL,
    )
    match = data_url_pattern.match(data_url)
    if not match:
        raise ValueError("Invalid Data URL format")

    parts = match.groupdict()
    parts["is_base64"] = bool(parts["base64"])
    return parts


def concat_interleaved_content(content: List[InterleavedContent]) -> InterleavedContent:
    """concatenate interleaved content into a single list. ensure that 'str's are converted to TextContentItem when in a list"""

    ret = []

    def _process(c):
        if isinstance(c, str):
            ret.append(TextContentItem(text=c))
        elif isinstance(c, list):
            for item in c:
                _process(item)
        else:
            ret.append(c)

    for c in content:
        _process(c)

    return ret


class EmbeddingIndex(ABC):
    @abstractmethod
    async def add_chunks(self, chunks: List[Chunk], embeddings: NDArray):
        raise NotImplementedError()

    @abstractmethod
    async def query(self, embedding: NDArray, k: int, score_threshold: float) -> QueryChunksResponse:
        raise NotImplementedError()

    @abstractmethod
    async def delete(self):
        raise NotImplementedError()


@dataclass
class VectorDBWithIndex:
    vector_db: VectorDB
    index: EmbeddingIndex
    inference_api: Api.inference

    async def insert_chunks(
        self,
        chunks: List[Chunk],
    ) -> None:
        embeddings_response = await self.inference_api.embeddings(
            self.vector_db.embedding_model, [x.content for x in chunks]
        )
        embeddings = np.array(embeddings_response.embeddings)

        await self.index.add_chunks(chunks, embeddings)

    async def query_chunks(
        self,
        query: InterleavedContent,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryChunksResponse:
        if params is None:
            params = {}
        k = params.get("max_chunks", 3)
        score_threshold = params.get("score_threshold", 0.0)

        query_str = interleaved_content_as_str(query)
        embeddings_response = await self.inference_api.embeddings(self.vector_db.embedding_model, [query_str])
        query_vector = np.array(embeddings_response.embeddings[0], dtype=np.float32)
        return await self.index.query(query_vector, k, score_threshold)
