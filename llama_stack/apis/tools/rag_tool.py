# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated, Protocol, runtime_checkable

from llama_stack.apis.common.content_types import URL, InterleavedContent
from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol
from llama_stack.schema_utils import json_schema_type, register_schema, webmethod


@json_schema_type
class RAGDocument(BaseModel):
    """
    A document to be used for document ingestion in the RAG Tool.

    :param document_id: The unique identifier for the document.
    :param content: The content of the document.
    :param mime_type: The MIME type of the document.
    :param metadata: Additional metadata for the document.
    """

    document_id: str
    content: InterleavedContent | URL
    mime_type: str | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


@json_schema_type
class RAGQueryResult(BaseModel):
    content: Optional[InterleavedContent] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


@json_schema_type
class RAGQueryGenerator(Enum):
    default = "default"
    llm = "llm"
    custom = "custom"


@json_schema_type
class DefaultRAGQueryGeneratorConfig(BaseModel):
    type: Literal["default"] = "default"
    separator: str = " "


@json_schema_type
class LLMRAGQueryGeneratorConfig(BaseModel):
    type: Literal["llm"] = "llm"
    model: str
    template: str


RAGQueryGeneratorConfig = Annotated[
    Union[
        DefaultRAGQueryGeneratorConfig,
        LLMRAGQueryGeneratorConfig,
    ],
    Field(discriminator="type"),
]
register_schema(RAGQueryGeneratorConfig, name="RAGQueryGeneratorConfig")


@json_schema_type
class RAGQueryConfig(BaseModel):
    # This config defines how a query is generated using the messages
    # for memory bank retrieval.
    query_generator_config: RAGQueryGeneratorConfig = Field(default=DefaultRAGQueryGeneratorConfig())
    max_tokens_in_context: int = 4096
    max_chunks: int = 5


@runtime_checkable
@trace_protocol
class RAGToolRuntime(Protocol):
    @webmethod(route="/tool-runtime/rag-tool/insert", method="POST")
    async def insert(
        self,
        documents: List[RAGDocument],
        vector_db_id: str,
        chunk_size_in_tokens: int = 512,
    ) -> None:
        """Index documents so they can be used by the RAG system"""
        ...

    @webmethod(route="/tool-runtime/rag-tool/query", method="POST")
    async def query(
        self,
        content: InterleavedContent,
        vector_db_ids: List[str],
        query_config: Optional[RAGQueryConfig] = None,
    ) -> RAGQueryResult:
        """Query the RAG system for context; typically invoked by the agent"""
        ...
