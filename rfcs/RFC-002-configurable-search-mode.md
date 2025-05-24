# Configurable Retrieval for RAG in Llama Stack

** Authors:**

* Red Hat: @varshaprasad96 @franciscojavierarceo

## Summary

This RFC proposes expanding the RAG capabilities in Llama Stack to support keyword search, hybrid search, and other
retrieval strategies through a backend configuration.

## Motivation

The benefits of pre-retrieval optimization through indexing have been well
studied ([1][rag-ref-1], [2][rag-ref-2], [3][rag-ref-3]) and it has been shown that keyword, vector, hybrid search, and other search strategies offer distinct benefits to
information retrieval for RAG. Enabling Llama Stack to easily configure different search modes, while abstracting the implementation
details offers significant value to Llama Stack users.

## Scope

### Goals:

1. Create a design to support different search modes across supported databases.

### Non-Goals:

1. Multi-mode stacking: We will focus on a single selectable search mode per database.

## Requirements:

To support `keyword` based searches (and consequently hybrid search), a query string is required. Therefore the `query`
method will need an additional parameter that is optional, we propose calling this new parameter `query_string`.

There are at least three different implementation options available:
1. Configure the Search Mode per application when registering the Vector Provider and applying the same mode for all
queries in the application.
2. Configure the Search Mode per query when querying the Vector Provider and allowing for different search modes for each
query in the application.
3. Create a new `keyword_search_io` provider and create a separate implementation for each provider.

A brief review of the pros and cons of the three options are provided in the table below.

### Implementation Evaluation

##### Option 1: Static search mode applied for all queries
*Pros*:
1. Easy to configure behind the scenes.
2. No need for additional APIs or breaking changes.

*Cons*:
1. Less flexible if users want to specify different queries for different options.
2. Retaining the VectorIO naming convention is not intuitive for the codebase.

##### Option 2: Dynamic search mode per query
*Pros*:
1. Allows queries to be flexible in their desired usage.
2. No need for additional APIs or breaking changes.

*Cons*:
1. Adds more complexity for API support.
2. Potentially exposes more challenges when users are debugging queries.
3. The database needs to be customized to store both embeddings and keywords. In certain implementations it could be a
memory overhead.
4. Complicates UX, the user should not have to care about which search mode they are using (e.g., users don’t configure
their search for OpenAI).
5. Unclear that there are use cases that this is a desirable parameter.
6. Retaining the VectorIO naming convention is not intuitive for the codebase.

##### Option 3: Separate Provider and API
*Pros*:
1. Allows maximum configuration.

*Cons*:
1. Larger implementation scope.
2. Requires providers to implement in 3 potential places (inline, remote, and keyword_search_io).
3. Generalization to hybrid search would logically warrant an additional provider implementation (hybrid_search_io).
4. Would duplicate a lot of boilerplate code (e.g., configuration of provider) across multiple databases depending on
their range of support (listed in reference).

## Recommendation and Proposal

Based on our review, the above pros and cons, and how other frameworks approach enabling hybrid and keyword search we
recommend:
Option 1: Allow Users to configure the Search Mode through a Provider Config field and an additional `query_string`
parameter in the API.

### Implementation Detail:

We would extend the RAGQueryConfig in the to accept a `mode` parameter:

```python
@json_schema_type
class RAGQueryConfig(BaseModel):
    # This config defines how a query is generated using the messages
    # for memory bank retrieval.
    query_generator_config: RAGQueryGeneratorConfig = Field(
        default=DefaultRAGQueryGeneratorConfig()
    )
    max_tokens_in_context: int = 4096
    max_chunks: int = 5
    mode: str
```

The Query API is modified to accept `query_string` and `mode` parameter within the `EmbeddingIndex` class:

```python
class EmbeddingIndex(ABC):
    @abstractmethod
    async def add_chunks(self, chunks: List[Chunk], embeddings: NDArray):
        raise NotImplementedError()

    @abstractmethod
    async def query(
        self,
        embedding: NDArray,
        query_string: Optional[str],
        k: int,
        score_threshold: float,
        mode: Optional[str],
    ) -> QueryChunksResponse:
        raise NotImplementedError()

    @abstractmethod
    async def delete(self):
        raise NotImplementedError()
```

Note: This change requires that all the implementations of the query API in the DBs need to be modified. It is also up
to the provider to ensure that a valid mode is provided.

The implementation after exposing the option to configure `mode` would look like the code below.

#### Step 1:

With querying the database directly:

```python
response = await sqlite_vec_index.query(
    embedding=query_embedding,
    query_string="",
    k=top_k,
    score_threshold=0.0,
    mode="vector",
)
```

With `RAGTool`:

```python
query_config = RAGQueryConfig(max_chunks=6, mode="vector").model_dump()
results = client.tool_runtime.rag_tool.query(
    vector_db_ids=[vector_db_id], content="what is torchtune", query_config=query_config
)
```

#### Step 2:

The eventual goal would be to make this change at the [DB registration][DB_registration] step, such that the user needs
to provide the search mode during query.

This change needs to be made in the [llama-stack-client][ls_client] and then propagated to the server.

### Benchmarking:

To evaluate the impact of configurable retrieval modes, we will benchmark the supported search strategies—keyword,
vector, and hybrid—across multiple vector database backends, as support for each is implemented.

#### References:

##### Databases and their supported configurations in Llama Stack as of April 11, 2025.

| Database                | Vector Search | Keyword Search | Hybrid Search |
|-------------------------|---------------|----------------|---------------|
| SQLite (inline)         | Yes           | Yes            | Yes           |
| FAISS (inline)          | Yes           | No             | No            |
| Chroma (inline, remote) | Yes           | Yes            | Yes           |
| Weaviate (remote)       | Yes           | Yes            | Yes           |
| Qdrant (remote)         | Yes           | Yes            | Yes           |
| PGVector (remote)       | Yes           | No             | No            |
| Milvus (inline, remote) | Yes           | Yes            | Yes           |

[rag-ref-1]: https://arxiv.org/pdf/2404.07220
[rag-ref-2]: https://arxiv.org/pdf/2312.10997
[rag-ref-3]: https://www.onlinescientificresearch.com/articles/optimizing-rag-with-hybrid-search-and-contextual-chunking.pdf
[DB_registration]: https://github.com/meta-llama/llama-stack-client-python/blob/b664564fe1c4771a7872286d0c2ac96c47816939/src/llama_stack_client/resources/vector_dbs.py#L105
[ls_client]: https://github.com/meta-llama/llama-stack-client-python/blob/main/src/llama_stack_client/resources/vector_dbs.py#L105
