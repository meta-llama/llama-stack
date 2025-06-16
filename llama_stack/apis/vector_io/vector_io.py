# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from llama_stack.apis.inference import InterleavedContent
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol
from llama_stack.schema_utils import json_schema_type, webmethod
from llama_stack.strong_typing.schema import register_schema


class Chunk(BaseModel):
    """
    A chunk of content that can be inserted into a vector database.
    :param content: The content of the chunk, which can be interleaved text, images, or other types.
    :param embedding: Optional embedding for the chunk. If not provided, it will be computed later.
    :param metadata: Metadata associated with the chunk, such as document ID, source, or other relevant information.
    """

    content: InterleavedContent
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


@json_schema_type
class QueryChunksResponse(BaseModel):
    chunks: list[Chunk]
    scores: list[float]


@json_schema_type
class VectorStoreObject(BaseModel):
    """OpenAI Vector Store object."""

    id: str
    object: str = "vector_store"
    created_at: int
    name: str | None = None
    usage_bytes: int = 0
    file_counts: dict[str, int] = Field(default_factory=dict)
    status: str = "completed"
    expires_after: dict[str, Any] | None = None
    expires_at: int | None = None
    last_active_at: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@json_schema_type
class VectorStoreCreateRequest(BaseModel):
    """Request to create a vector store."""

    name: str | None = None
    file_ids: list[str] = Field(default_factory=list)
    expires_after: dict[str, Any] | None = None
    chunking_strategy: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@json_schema_type
class VectorStoreModifyRequest(BaseModel):
    """Request to modify a vector store."""

    name: str | None = None
    expires_after: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


@json_schema_type
class VectorStoreListResponse(BaseModel):
    """Response from listing vector stores."""

    object: str = "list"
    data: list[VectorStoreObject]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False


@json_schema_type
class VectorStoreSearchRequest(BaseModel):
    """Request to search a vector store."""

    query: str | list[str]
    filters: dict[str, Any] | None = None
    max_num_results: int = 10
    ranking_options: dict[str, Any] | None = None
    rewrite_query: bool = False


@json_schema_type
class VectorStoreContent(BaseModel):
    type: Literal["text"]
    text: str


@json_schema_type
class VectorStoreSearchResponse(BaseModel):
    """Response from searching a vector store."""

    file_id: str
    filename: str
    score: float
    attributes: dict[str, str | float | bool] | None = None
    content: list[VectorStoreContent]


@json_schema_type
class VectorStoreSearchResponsePage(BaseModel):
    """Response from searching a vector store."""

    object: str = "vector_store.search_results.page"
    search_query: str
    data: list[VectorStoreSearchResponse]
    has_more: bool = False
    next_page: str | None = None


@json_schema_type
class VectorStoreDeleteResponse(BaseModel):
    """Response from deleting a vector store."""

    id: str
    object: str = "vector_store.deleted"
    deleted: bool = True


@json_schema_type
class VectorStoreChunkingStrategyAuto(BaseModel):
    type: Literal["auto"] = "auto"


@json_schema_type
class VectorStoreChunkingStrategyStaticConfig(BaseModel):
    chunk_overlap_tokens: int = 400
    max_chunk_size_tokens: int = Field(800, ge=100, le=4096)


@json_schema_type
class VectorStoreChunkingStrategyStatic(BaseModel):
    type: Literal["static"] = "static"
    static: VectorStoreChunkingStrategyStaticConfig


VectorStoreChunkingStrategy = Annotated[
    VectorStoreChunkingStrategyAuto | VectorStoreChunkingStrategyStatic, Field(discriminator="type")
]
register_schema(VectorStoreChunkingStrategy, name="VectorStoreChunkingStrategy")


@json_schema_type
class VectorStoreFileLastError(BaseModel):
    code: Literal["server_error"] | Literal["rate_limit_exceeded"]
    message: str


@json_schema_type
class VectorStoreFileObject(BaseModel):
    """OpenAI Vector Store File object."""

    id: str
    object: str = "vector_store.file"
    attributes: dict[str, Any] = Field(default_factory=dict)
    chunking_strategy: VectorStoreChunkingStrategy
    created_at: int
    last_error: VectorStoreFileLastError | None = None
    status: Literal["completed"] | Literal["in_progress"] | Literal["cancelled"] | Literal["failed"]
    usage_bytes: int = 0
    vector_store_id: str


class VectorDBStore(Protocol):
    def get_vector_db(self, vector_db_id: str) -> VectorDB | None: ...


@runtime_checkable
@trace_protocol
class VectorIO(Protocol):
    vector_db_store: VectorDBStore | None = None

    # this will just block now until chunks are inserted, but it should
    # probably return a Job instance which can be polled for completion
    @webmethod(route="/vector-io/insert", method="POST")
    async def insert_chunks(
        self,
        vector_db_id: str,
        chunks: list[Chunk],
        ttl_seconds: int | None = None,
    ) -> None:
        """Insert chunks into a vector database.

        :param vector_db_id: The identifier of the vector database to insert the chunks into.
        :param chunks: The chunks to insert. Each `Chunk` should contain content which can be interleaved text, images, or other types.
            `metadata`: `dict[str, Any]` and `embedding`: `List[float]` are optional.
            If `metadata` is provided, you configure how Llama Stack formats the chunk during generation.
            If `embedding` is not provided, it will be computed later.
        :param ttl_seconds: The time to live of the chunks.
        """
        ...

    @webmethod(route="/vector-io/query", method="POST")
    async def query_chunks(
        self,
        vector_db_id: str,
        query: InterleavedContent,
        params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        """Query chunks from a vector database.

        :param vector_db_id: The identifier of the vector database to query.
        :param query: The query to search for.
        :param params: The parameters of the query.
        :returns: A QueryChunksResponse.
        """
        ...

    # OpenAI Vector Stores API endpoints
    @webmethod(route="/openai/v1/vector_stores", method="POST")
    async def openai_create_vector_store(
        self,
        name: str,
        file_ids: list[str] | None = None,
        expires_after: dict[str, Any] | None = None,
        chunking_strategy: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        embedding_model: str | None = None,
        embedding_dimension: int | None = 384,
        provider_id: str | None = None,
        provider_vector_db_id: str | None = None,
    ) -> VectorStoreObject:
        """Creates a vector store.

        :param name: A name for the vector store.
        :param file_ids: A list of File IDs that the vector store should use. Useful for tools like `file_search` that can access files.
        :param expires_after: The expiration policy for a vector store.
        :param chunking_strategy: The chunking strategy used to chunk the file(s). If not set, will use the `auto` strategy.
        :param metadata: Set of 16 key-value pairs that can be attached to an object.
        :param embedding_model: The embedding model to use for this vector store.
        :param embedding_dimension: The dimension of the embedding vectors (default: 384).
        :param provider_id: The ID of the provider to use for this vector store.
        :param provider_vector_db_id: The provider-specific vector database ID.
        :returns: A VectorStoreObject representing the created vector store.
        """
        ...

    @webmethod(route="/openai/v1/vector_stores", method="GET")
    async def openai_list_vector_stores(
        self,
        limit: int | None = 20,
        order: str | None = "desc",
        after: str | None = None,
        before: str | None = None,
    ) -> VectorStoreListResponse:
        """Returns a list of vector stores.

        :param limit: A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20.
        :param order: Sort order by the `created_at` timestamp of the objects. `asc` for ascending order and `desc` for descending order.
        :param after: A cursor for use in pagination. `after` is an object ID that defines your place in the list.
        :param before: A cursor for use in pagination. `before` is an object ID that defines your place in the list.
        :returns: A VectorStoreListResponse containing the list of vector stores.
        """
        ...

    @webmethod(route="/openai/v1/vector_stores/{vector_store_id}", method="GET")
    async def openai_retrieve_vector_store(
        self,
        vector_store_id: str,
    ) -> VectorStoreObject:
        """Retrieves a vector store.

        :param vector_store_id: The ID of the vector store to retrieve.
        :returns: A VectorStoreObject representing the vector store.
        """
        ...

    @webmethod(route="/openai/v1/vector_stores/{vector_store_id}", method="POST")
    async def openai_update_vector_store(
        self,
        vector_store_id: str,
        name: str | None = None,
        expires_after: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VectorStoreObject:
        """Updates a vector store.

        :param vector_store_id: The ID of the vector store to update.
        :param name: The name of the vector store.
        :param expires_after: The expiration policy for a vector store.
        :param metadata: Set of 16 key-value pairs that can be attached to an object.
        :returns: A VectorStoreObject representing the updated vector store.
        """
        ...

    @webmethod(route="/openai/v1/vector_stores/{vector_store_id}", method="DELETE")
    async def openai_delete_vector_store(
        self,
        vector_store_id: str,
    ) -> VectorStoreDeleteResponse:
        """Delete a vector store.

        :param vector_store_id: The ID of the vector store to delete.
        :returns: A VectorStoreDeleteResponse indicating the deletion status.
        """
        ...

    @webmethod(route="/openai/v1/vector_stores/{vector_store_id}/search", method="POST")
    async def openai_search_vector_store(
        self,
        vector_store_id: str,
        query: str | list[str],
        filters: dict[str, Any] | None = None,
        max_num_results: int | None = 10,
        ranking_options: dict[str, Any] | None = None,
        rewrite_query: bool | None = False,
    ) -> VectorStoreSearchResponsePage:
        """Search for chunks in a vector store.

        Searches a vector store for relevant chunks based on a query and optional file attribute filters.

        :param vector_store_id: The ID of the vector store to search.
        :param query: The query string or array for performing the search.
        :param filters: Filters based on file attributes to narrow the search results.
        :param max_num_results: Maximum number of results to return (1 to 50 inclusive, default 10).
        :param ranking_options: Ranking options for fine-tuning the search results.
        :param rewrite_query: Whether to rewrite the natural language query for vector search (default false)
        :returns: A VectorStoreSearchResponse containing the search results.
        """
        ...

    @webmethod(route="/openai/v1/vector_stores/{vector_store_id}/files", method="POST")
    async def openai_attach_file_to_vector_store(
        self,
        vector_store_id: str,
        file_id: str,
        attributes: dict[str, Any] | None = None,
        chunking_strategy: VectorStoreChunkingStrategy | None = None,
    ) -> VectorStoreFileObject:
        """Attach a file to a vector store.

        :param vector_store_id: The ID of the vector store to attach the file to.
        :param file_id: The ID of the file to attach to the vector store.
        :param attributes: The key-value attributes stored with the file, which can be used for filtering.
        :param chunking_strategy: The chunking strategy to use for the file.
        :returns: A VectorStoreFileObject representing the attached file.
        """
        ...
