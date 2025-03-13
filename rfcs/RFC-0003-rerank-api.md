# RFC: Llama-Stack Reranking for RAG Workflows

**Status:** Draft
**Author:** Kevin Cogan
**Start Date:** 2025-02-24

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Problem Statement](#problem-statement)
4. [Proposed Reranking Solution](#proposed-reranking-solution)
   4.1. [Extended API Endpoints](#41-extended-api-endpoints)
   &nbsp;&nbsp;&nbsp;&nbsp;4.1.1. [Query Endpoint](#411-query-endpoint)
   &nbsp;&nbsp;&nbsp;&nbsp;4.1.2. [Dedicated Rerank Endpoint](#412-dedicated-rerank-endpoint)
   4.2. [Data Models and Schemas](#42-data-models-and-schemas)
   4.3. [Implementation in the RAG Runtime](#43-implementation-in-the-rag-runtime)
   &nbsp;&nbsp;&nbsp;&nbsp;4.3.1. [Inline Reranking Integration](#431-inline-reranking-integration)
   &nbsp;&nbsp;&nbsp;&nbsp;4.3.2. [Reranker Service](#432-reranker-service)
   4.4. [Example Configuration and Request](#44-example-configuration-and-request)
5. [API Design Overview](#api-design-overview)
6. [Considerations and Tradeoffs](#6-considerations-and-tradeoffs)
7. [Conclusion](#7-conclusion)
8. [Approvals](#8-approvals)

## Abstract

This RFC proposes an enhancement to the Llama-Stack Retrieval-Augmented Generation (RAG) system through a configurable reranking component. Many enterprise users rely on legacy keyword search systems that already have significant investments in content synchronization and indexing. In these cases, re-ranking can improve accuracy by refining search results without requiring a full transition to a vector-based retrieval system. By incorporating an additional scoring step—using either a remote inference API or a self-hosted model—the system enhances document retrieval, providing more precise context for downstream tasks. Users have the flexibility to enable or disable reranking and to select a reranker from remote providers (e.g., LlamaRank, Voyage AI, Cohere) or self-hosted models (e.g., sentence-transformers, LLM-based inference). Additionally, telemetry updates are integrated to capture and report reranking metrics for enhanced observability and performance tuning.
By incorporating an additional scoring step—using either a remote inference API or a self-hosted model—the system enhances document retrieval, providing more precise context for downstream tasks. Users have the flexibility to enable or disable reranking and to select a reranker from remote providers (e.g., LlamaRank, Voyage AI, Cohere) or self-hosted models (e.g., sentence-transformers, LLM-based inference). Additionally, telemetry updates are integrated to capture and report reranking metrics for enhanced observability and performance tuning.

## Introduction

Current RAG implementations use embedding-based similarity search to retrieve document candidates; however, the preliminary ordering can be suboptimal for ambiguous or complex queries. For enterprise users who rely on keyword-based search systems, re-ranking can be especially impactful, as it enhances accuracy without requiring a full migration to vector search. This document outlines an approach that provides both API-based reranking and inline reranking, ensuring seamless integration with existing retrieval systems while emphasizing configurability, ease of implementation, and robust telemetry reporting.

## Problem Statement

Existing RAG systems efficiently index and retrieve document chunks from vector stores, but they often lack a mechanism to refine initial results. This can lead to suboptimal context for LLMs and hinder overall performance. The case for re-ranking is especially strong for enterprise users relying on legacy keyword search systems, where significant investments have already been made in content synchronization and indexing. In these environments, re-ranking can substantially improve accuracy by refining outputs from established search infrastructure. While new vector stores using state-of-the-art dense models also benefit from re-ranking, the improvements tend to be less pronounced and may not justify the additional complexity and latency. Moreover, different operational needs mean that some users prefer a managed API solution, while others require inline control for low latency or data privacy.

## Proposed Reranking Solution

![Figure 1: Model Life Cycle](../docs/resources/rerank_api_flowchart.png)

## 4.1. Extended API Endpoints

### 4.1.1. Query Endpoint

The `/tool-runtime/rag-tool/query` endpoint will be updated to accept three additional parameters:

- `rerank_strategy` (RerankingStrategy): Determines the ranking strategy to be applied. Options include:
  - `NONE` - No reranking applied.
  - `DEFAULT` (default) - Standard reranking based on computed relevance scores.
  - `BOOST` - Boost-based ranking to emphasize specific documents.
  - `HYBRID` - Combines multiple ranking methods for improved results.
  - `LLM_RERANK` - Uses an LLM-based model for reranking.
  - etc… - More can be added when needed.
- `reranker_model_id` (string): Specifies the reranking provider or model (e.g., `"sentence-transformers"` or self-hosted models).
- `rerank_config` (`Optional[RAGRerankConfig]`): Configures additional options for the reranking process (e.g. `api_url`, `api_key`).

The updated endpoint interface is as follows:

```python
@runtime_checkable
@trace_protocol
class RAGToolRuntime(Protocol):
    @webmethod(route="/tool-runtime/rag-tool/query", method="POST")
    async def query(
        self,
        content: InterleavedContent,
        vector_db_ids: List[str],
        query_config: Optional[RAGQueryConfig] = None,
        rerank_strategy: RankingStrategy = RankingStrategy.DEFAULT,
        reranker_model_id: str = "my_model_id",
        rerank_config: Optional[RAGRerankConfig] = None,
    ) -> RAGQueryResult: ...
```

> <sub>**Note:** Note: When rerank_strategy is not None, the service will invoke the reranking process using the specified reranker_model_id and additional options defined in rerank_config.

### 4.1.2. Dedicated Rerank Endpoint

A new endpoint, `/tool-runtime/rag-tool/rerank`, is introduced. It accepts the following parameters:

- **`query`** (_InterleavedContent_): The input search query.
- **`retrieved_docs`** (_List[RAGDocument]_): A list of retrieved documents.
- `rerank_strategy` (RerankingStrategy): Determines the ranking strategy to be applied. Options include:
  - `NONE` - No reranking applied.
  - `DEFAULT` (default) - Standard reranking based on computed relevance scores.
  - `BOOST` - Boost-based ranking to emphasize specific documents.
  - `HYBRID` - Combines multiple ranking methods for improved results.
  - `LLM_RERANK` - Uses an LLM-based model for reranking.
  - etc… - More can be added when needed.
- **`reranker_model_id`** (_string_): Identifier of the reranker model.
- **`top_k`** (_integer_): The number of top documents to return.
- **`rerank_config`** (_Optional[RAGRerankConfig]_): Configures additional options for the reranking process (e.g. `api_url`, `api_key`).

Below is an example implementation of the endpoint:

```python
@webmethod(route="/tool-runtime/rag-tool/rerank", method="POST")
async def rerank(
    self,
    query: InterleavedContent,
    retrieved_docs: List[RAGDocument],
    top_k: int = 5,
    rerank_strategy: RerankingStrategy = RerankingStrategy.DEFAULT,
    reranker_model_id: Optional[str] = None,
    rerank_config: Optional[RAGRerankConfig] = None,
) -> RerankResponse:
    """Re-rank retrieved documents based on relevance"""
    ...
```

## 4.2. Data Models and Schemas

The following Pydantic schemas define the data models that support the reranking process. They ensure that the input and output data structures conform to expected types and constraints.

- **RerankedDocument**:
  Represents an individual document after the reranking operation. It includes the document's original index and a computed relevance score.

- **RerankResponse**:
  Wraps the results from the reranking process. It includes a list of reranked documents along with optional metadata for any additional context.

- **RAGRerankConfig**:
  Provides optional configuration settings for external services used during reranking.

- **RerankingStrategy**
  Defines the different strategies available for reranking, ranging from basic ranking to more advanced techniques leveraging LLMs and hybrid approaches.

Below is the updated schema definition:

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class RerankedDocument(BaseModel):
    """Represents a single document after reranking.

    Attributes:
        index (int): The original position of the document.
        relevance_score (float): The computed relevance score.
    """

    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    """Defines the response structure for the rerank endpoint.

    Attributes:
        reranked_documents (List[RerankedDocument]): The list of
        documents sorted by relevance.
        metadata (Dict[str, Any]): Additional metadata related to the
        reranking process.
    """

    reranked_documents: List[RerankedDocument]
    metadata: Dict[str, Any] = Field(default_factory=dict)


@json_schema_type
class RAGRerankConfig(BaseModel):
    """Configuration settings for the reranking service.

    Attributes:
        api_url (Optional[str]): The API endpoint for the external
        service (if applicable).
        api_key (Optional[str]): The API key for authenticating with the
        external service.
    """

    api_url: Optional[str] = None
    api_key: Optional[str] = None


class RerankingStrategy(Enum):
    """Defines different strategies for reranking documents.

    Attributes:
        NONE (str): No reranking is applied.
        RERANK (str): Standard reranking based on relevance scores.
        BOOST (str): Boosting-based ranking for emphasizing certain documents.
        LLM_RERANK (str): Leverages an LLM-based model for reranking.
        HYBRID (str): Combines multiple ranking methods for improved results.
    """

    NONE = "none"
    DEFAULT = "default"
    BOOST = "boost"
    LLM_RERANK = "llm_rerank"
    HYBRID = "hybrid"
```

## 4.3. Implementation in the RAG Runtime

### 4.3.1. Inline Reranking Integration

Within the RAG runtime (e.g., in `/llama_stack/providers/inline/tool_runtime/rag/memory.py`), the `query` method is extended to support inline reranking. When the `rerank` flag is enabled, the system calls a dedicated reranking endpoint via the `RerankerService` to re-score the initially retrieved document chunks.

Below is an improved example of the integration:

```python
class MemoryToolRuntimeImpl(ToolsProtocolPrivate, ToolRuntime, RAGToolRuntime):

    async def query(
        self,
        content: InterleavedContent,
        vector_db_ids: List[str],
        query_config: Optional[RAGQueryConfig] = None,
        rerank_strategy: RerankingStrategy = RerankingStrategy.DEFAULT,
        reranker_model_id: str = "my_model_id",
        rerank_config: Optional[RAGRerankConfig] = None,
    ) -> RAGQueryResult:

        # ... [initial retrieval logic that produces `chunks` and `scores`] ...

        if rerank_strategy:

            # Call RerankerService to obtain refined relevance scores.

            # Note: The reranker service uses the query, the retrieved document chunks, and any additional config.

            reranked_results = await RerankerService.rerank_documents(
                query=content,
                documents=list(chunks),
                rerank_strategy=rerank_strategy,
                config=rerank_config,
            )

            # Build a mapping from the original chunk index to its new reranked score.

            index_to_score = {
                doc.index: doc.relevance_score
                for doc in reranked_results.reranked_documents
            }

            # Combine each chunk with its corresponding reranked score (defaulting to 0.0 if absent).

            sort_data = [
                (chunk, index_to_score.get(i, 0.0)) for i, chunk in enumerate(chunks)
            ]

        else:

            # Use the original scores when reranking is not enabled.

            sort_data = list(zip(chunks, scores))

        # Sort chunks by score in descending order and extract the sorted chunks.

        reranked_chunks = [
            chunk
            for chunk, _ in sorted(sort_data, key=lambda pair: pair[1], reverse=True)
        ]

    # ... [further processing to build and return a RAGQueryResult] ...
```

> <sub>**Note:** Note: When a rerank_strategy is specified, an additional reranking process is applied. If the rerank_config includes a URL or API key, the system will use the external reranker; otherwise, it defaults to the local reranker. This design clearly separates enabling reranking from selecting the provider.

### 4.3.2. Reranker Service

The Reranker Service is responsible for reordering document chunks based on their relevance to a given query. Its design is modular and composed of five main parts:

#### RerankerProvider

The `RerankerProvider` class serves as an abstract base class (ABC) that defines a common interface for all reranker implementations. This ensures that any subclass (such as `LocalReranker` and `ExternalReranker`) must implement the `compute_scores` method.

```python
from abc import ABC, abstractmethod
from typing import List, Any
import numpy as np


class RerankerProvider(ABC):
    @abstractmethod
    async def compute_scores(self, query: str, chunks: List[Any]) -> np.ndarray:
        """Compute relevance scores for a list of document chunks."""
        pass
```

#### LocalReranker

Reranks query–document pairs based on a specified strategy. It requires a `model_id` and a `rerank_strategy`. If `model_id` is missing, a `ValueError` is raised.

##### Initialization

- Call `LocalReranker(model_id, rerank_strategy)` to create an instance.
- The constructor immediately calls `_initialize_model()` asynchronously to set up the model.
  - If `model_id` is not provided, a ValueError is raised.
  - If `rerank_strategy == RerankingStrategy.LLM_RERANK`, the component returns the provided model_id as is (placeholder behavior).
  - Otherwise, it assumes model_id corresponds to a cross-encoder or other model that supports `.predict()`.

##### Scoring

- Call compute_scores(query, chunks).
- The method pairs the query with each chunk’s content to form (query, content) pairs.
- The exact post-processing depends on rerank_strategy:
  - `NONE`: Returns zeros of length chunks.
  - `DEFAULT`: Returns the raw scores from predict().
  - `BOOST`: Placeholder for boosted scores (currently the same as raw).
  - `HYBRID`: Placeholder for a hybrid approach (currently the same as raw).
  - `LLM_RERANK`: Placeholder for LLM-based logic.

```python
class LocalReranker(RerankerProvider):

    def __init__(
        self,
        model_id: Optional[str] = None,
        rerank_strategy: RerankingStrategy = RerankingStrategy.DEFAULT,
    ) -> None:

        self.model_id = model_id
        self.rerank_strategy = rerank_strategy
        self.model = asyncio.run(self._initialize_model())

    async def _initialize_model(self) -> Any:
        if not self.model_id:
            raise ValueError(
                "No model_id provided, but a model is required for this reranking strategy."
            )

        if self.rerank_strategy == RerankingStrategy.LLM_RERANK:
            # Placeholder for an LLM-based model loader
            return self.model_id

        # Default: use model_id for cross-encoder/embedding-based model loading
        return self.model_id

    async def compute_scores(self, query: str, chunks: List[Any]) -> np.ndarray:
        if self.rerank_strategy == RerankingStrategy.NONE:
            return np.zeros(len(chunks))

        # Create (query, content) pairs from chunks
        pairs = [(query, chunk.content) for chunk in chunks]

        # Compute scores in a non-blocking way
        scores = await asyncio.to_thread(self.model.predict, pairs)

        if self.rerank_strategy == RerankingStrategy.DEFAULT:
            return scores

        elif self.rerank_strategy == RerankingStrategy.BOOST:
            # Placeholder: apply a boost factor to scores
            return scores

        elif self.rerank_strategy == RerankingStrategy.HYBRID:
            # Placeholder: combine cross-encoder scores with embedding-based scores
            return scores

        elif self.rerank_strategy == RerankingStrategy.LLM_RERANK:
            # Placeholder: use an LLM to compute scores
            return scores

        else:
            raise ValueError(f"Unknown reranking strategy: {self.rerank_strategy}")
```

#### ExternalReranker

For scenarios where reranking is handled externally, this provider sends the search query and document chunk contents to a specified API endpoint. It constructs a JSON payload—including the query, document contents, API key, and any extra parameters—and performs an asynchronous HTTP POST request to obtain relevance scores. If the API call fails (due to connectivity issues, HTTP errors, or other unexpected issues), it raises a `RuntimeError` with a descriptive error message.

```python
class ExternalReranker(RerankerProvider):
    def __init__(
        self,
        api_url: str,
        api_key: str,
        headers: Optional[dict] = None,
        timeout: float = 5.0,
        **kwargs,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.headers = headers or {}
        self.timeout = timeout
        # Store any additional parameters for later use in the request payload.
        self.extra_params = kwargs

    async def compute_scores(self, query: str, chunks: List[Any]) -> np.ndarray:
        # Build the payload including the api_key and any extra parameters.
        payload = {
            "query": query,
            "documents": [chunk.content for chunk in chunks],
            "api_key": self.api_key,
            **self.extra_params,  # Include any additional unlisted variables
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.api_url, json=payload, headers=self.headers
                )
                response.raise_for_status()
                data = response.json()
                scores = np.array(data.get("scores", []))
                return scores
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error while calling external reranker: {e}"
            ) from e
```

#### RerankerProviderFactory

This factory class determines whether to instantiate a local or external reranker based on the provided input parameters. If an `api_url` is supplied, it creates an `ExternalReranker` instance with the given API URL, headers, and timeout. If `api_url` is missing, a `ValueError` is raised. If no external provider is specified, the factory defaults to creating a `LocalReranker`, optionally using a provided model identifier.

```python
class RerankerProviderFactory:
    @staticmethod
    def get_provider(**kwargs) -> RerankerProvider:
        api_url = kwargs.get("api_url", False)
        if not api_url:
            raise ValueError("api_url must be provided for the external provider")
        return ExternalReranker(
            api_url=api_url,
            headers=kwargs.get("headers", {}),
            timeout=kwargs.get("timeout", 60.0),
        )
        # Default to local provider
        return LocalReranker(model=kwargs.get("model"))
```

#### RerankerService

The `RerankerService` is the core component that orchestrates the entire document reranking process. It performs the following steps:

- #### Relevance Score Computation:

  It begins by computing relevance scores for each document chunk using a configured provider (either local or external).

- #### Metric Calculation:

  The service then calculates key statistical metrics from the scores:

  - **Mean Score**: The average relevance score.
  - **Standard Deviation**: The variability of the scores.
  - **Score Gap**: The difference between the highest and the second-highest scores.

  > <sub> **Note**: Metrics can be added or removed based on the specific insights or data points you want to collect.

#### Telemetry Logging

To ensure observability, telemetry data—including raw scores, execution time, and the calculated metrics—is logged via tracing utilities.

#### Reranked Document Construction

It builds a list of reranked documents, where each document is annotated with an index and its relevance score. The list is then sorted in descending order of relevance.

#### Reranked Response and Document Metadata

Finally, the service returns a detailed response that includes both the sorted reranked documents and associated metadata.

For convenience, a static method `rerank_documents` is provided. This method dynamically selects the appropriate provider based on configuration parameters and initiates the reranking process, offering a simple entry point for external callers.

```python
class RerankerService:
    """
    A service for reranking document chunks based on their relevance to a
    query. Supports both local models and external API-based rerankers.
    """

    def __init__(self, provider: RerankerProvider) -> None:
        self.provider = provider

    async def _compute_scores(self, query: str, chunks: List[Any]) -> np.ndarray:
        """Computes relevance scores for document chunks using the
        assigned provider."""
        return await self.provider.compute_scores(query, chunks)

    def _calculate_metrics(self, scores: np.ndarray) -> Dict[str, float]:
        """Calculates statistical metrics for the computed relevance
        scores."""
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        score_gap = (
            float(np.max(scores) - np.partition(scores, -2)[-2])
            if len(scores) > 1
            else 0.0
        )
        return {
            "mean_score": mean_score,
            "std_score": std_score,
            "score_gap": score_gap,
        }

    def _log_telemetry(
        self,
        scores: np.ndarray,
        chunks: List[Any],
        exec_time_ms: float,
        metrics: Dict[str, float],
    ) -> None:
        """Logs telemetry data for reranking performance and
        analytics."""
        current_span = get_current_span()
        if current_span:
            current_span.set_attribute("reranker.raw_scores", serialize_value(scores))
            current_span.set_attribute(
                "reranker.top_score",
                serialize_value(float(np.max(scores)) if scores.size > 0 else None),
            )
            current_span.set_attribute(
                "reranker.chunk_count", serialize_value(len(chunks))
            )
            current_span.set_attribute(
                "reranker.execution_time_ms", serialize_value(exec_time_ms)
            )
            current_span.set_attribute(
                "reranker.mean_score", serialize_value(metrics["mean_score"])
            )
            current_span.set_attribute(
                "reranker.std_score", serialize_value(metrics["std_score"])
            )
            current_span.set_attribute(
                "reranker.score_gap", serialize_value(metrics["score_gap"])
            )

    def _build_reranked_documents(self, scores: np.ndarray) -> List[RerankedDocument]:
        """Creates and sorts reranked documents based on computed
        scores."""
        return sorted(
            [
                RerankedDocument(index=i, relevance_score=score)
                for i, score in enumerate(scores)
            ],
            key=lambda doc: doc.relevance_score,
            reverse=True,
        )

    async def _rerank_documents(self, query: str, chunks: List[Any]) -> RerankResponse:
        """
        Reranks document chunks based on their computed relevance scores.

        Parameters:
            query (str): The search query.
            chunks (List[Any]): List of document chunks to be reranked.

        Returns:
            RerankResponse: Contains reranked documents and metadata.
        """
        if not chunks:
            return RerankResponse(reranked_documents=[], metadata={"query": query})

        start_time = time.time()
        scores = await self._compute_scores(query, chunks)
        exec_time_ms = (time.time() - start_time) * 1000

        metrics = self._calculate_metrics(scores)
        self._log_telemetry(scores, chunks, exec_time_ms, metrics)

        return RerankResponse(
            reranked_documents=self._build_reranked_documents(scores),
            metadata={
                "query": query,
                "total_chunks": len(chunks),
                "raw_scores": scores,
                "execution_time_ms": exec_time_ms,
                **metrics,
            },
        )

    @staticmethod
    async def rerank_documents(
        query: str,
        chunks: List[Any],
        rerank_strategy: RerankingStrategy,
        **provider_kwargs
    ) -> RerankResponse:
        """
        Selects a reranker (local or external) and processes document
        reranking.

        Parameters:
            query (str): The search query.
            chunks (List[Any]): List of document chunks.
                rerank_strategy:the reranking option to execute.
            provider_kwargs (dict): Additional parameters for the provider (e.g., 'api_url' for external providers).

        Returns:
            RerankResponse: The reranked documents with associated
            metadata.
        """
        provider = RerankerProviderFactory.get_provider(
            rerank_strategy, **provider_kwargs
        )
        return await RerankerService(provider)._rerank_documents(query, chunks)
```

## 4.4. Example Configuration and Request

Below is an example of how an inline reranking configuration might be specified:

```python
agent_config = AgentConfig(
    model="meta-llama/Llama-3.2-3B-Instruct",
    instructions="Your specific instructions here",
    enable_session_persistence=True,
    max_infer_iters=3,
    toolgroups=[
        {
            "name": "RAG Retrieval Group",
            "args": {
                "vector_db_ids": ["db1", "db2"],
                "top_k": 5,
                "query_config": {
                    "max_tokens_in_context": 512,
                    "max_chunks": 10,
                    "query_generator_config": {"type": "simple", "separator": " "},
                },
                "rerank_strategy": "DEFAULT",
                "reranker_model_id": "model_name/model_path",
                "rerank_config": {
                    "api_url": "https://api.together.xyz/v1",
                    "api_key": "API_KEY",
                },
            },
        }
    ],
    tool_choice="auto",
)
```

And an example API call using cURL:

```bash
curl -X POST "http://localhost:8000/tool-runtime/rag-tool/rerank" \

     -H "Content-Type: application/json" \

     -d '{

         "query": "Find relevant documents for query text.",

         "retrieved_docs": [/* List of RAGDocument objects */],

         "top_k": 5,

         "rerank_strategy": "DEFAULT",

         "reranker_model_id": "LlamaRank",

         "rerank_config: {

                      "api_url": "https://api.together.xyz/v1",

                      "api_key": "API_KEY",

                 }

     }'
```

## API Design Overview

### 5.1. Extended Query Endpoint

**Endpoint:** `/tool-runtime/rag-tool/query`
**Method:** `POST`

#### Parameters:

- `content`: Input query content.
- `vector_db_ids`: List of vector database identifiers.
- `query_config`: Optional dictionary query configuration.
- `rerank_strategy` (RerankingStrategy): Determines the ranking strategy to be applied. Options include:
  - `NONE` - No reranking applied.
  - `DEFAULT` (default) - Standard reranking based on computed relevance scores.
  - `BOOST` - Boost-based ranking to emphasize specific documents.
  - `HYBRID` - Combines multiple ranking methods for improved results.
  - `LLM_RERANK` - Uses an LLM-based model for reranking.
  - etc… - More can be added when needed.
- `reranker_model_id`: String identifier for the reranking model.
- `rerank_config`: Optional dictionary rerank configuration.
  - `api_url`: URL for the external reranking service.
  - `api_key`: Authentication key for the external service.

### 5.2. Dedicated Rerank Endpoint

**Endpoint:** `/tool-runtime/rag-tool/rerank`
**Method:** `POST`

#### Parameters:

- `query`: Search query content.
- `retrieved_docs`: List of initially retrieved documents.
- `top_k`: Number of top documents to return.
- `rerank_strategy` (RerankingStrategy): Determines the ranking strategy to be applied. Options include:
  - `NONE` - No reranking applied.
  - `DEFAULT` (default) - Standard reranking based on computed relevance scores.
  - `BOOST` - Boost-based ranking to emphasize specific documents.
  - `HYBRID` - Combines multiple ranking methods for improved results.
  - `LLM_RERANK` - Uses an LLM-based model for reranking.
  - etc… - More can be added when needed.
- `reranker_model_id`: Identifier for the reranking model.
- `rerank_config`: Optional dictionary rerank configuration.
  - `api_url`: URL for the external reranking service.
  - `api_key`: Authentication key for the external service.

## 6. Considerations and Tradeoffs

#### Flexibility vs. Complexity

- **Flexibility**: The design allows users to choose between local and external reranking solutions and even swap out the default model.
- **Complexity**: This added flexibility introduces extra configuration options, requiring careful management of different providers and error-handling scenarios.

#### Performance vs. Latency

- **Performance Improvement**: Reranking can enhance document relevance, providing more precise context for downstream tasks.
- **Latency Overhead**: The additional scoring step can introduce extra latency, especially when using external API calls or complex models.

#### Observability vs. Implementation Overhead

- **Observability**: Detailed telemetry (e.g., raw scores, computed metrics, execution time) improves debugging and performance tuning.
- **Overhead**: Collecting and processing this telemetry data can add to system overhead and complexity.

#### Local vs. External Provider Tradeoffs

- **Local Provider**: Offers lower latency and greater control, suitable for environments with strict data privacy or low latency requirements.
- **External Provider**: Enables managed, scalable inference but depends on network connectivity and may have higher operational costs or variability in response times.

#### Legal and Intellectual Property Risks

- **Legal Uncertainty:**
  Some reranking models, such as cross-encoders trained on MS MARCO, may be released under permissive licenses (e.g., Apache 2.0) but are trained on datasets with non-commercial use restrictions. This creates ambiguity regarding their legal use in enterprise or commercial environments.

- **Risk of Default Model Selection:**
  Given potential IP concerns, it may not be advisable to provide a default reranking model. Instead:

  - **User Selection:** Users should explicitly select their own reranking model to ensure compliance with their legal and licensing policies.
  - **Model-Agnostic System:** The system should remain model-agnostic, allowing integration with vetted rerankers that meet organizational requirements.
  - **Legal Disclaimer:** A legal disclaimer should be included, clarifying that users bear responsibility for verifying model licensing.

- **Alternative Approaches:**
  Some organizations, such as InstructLab, default to Granite Embeddings to ensure clearer legal standing.

By not enforcing a default reranker, this approach shifts responsibility to users, allowing them to make informed decisions based on their legal and compliance needs.

## 7. Conclusion

The proposed reranking mechanism addresses the shortcomings of traditional document retrieval by refining initial results to deliver more relevant and precise context for downstream tasks. By offering both external API and local inference options, the solution provides a flexible and scalable approach that can be tailored to diverse operational scenarios. With defined API endpoints and telemetry, this design lays the foundation for iterative enhancements and further collaboration, ensuring the system can evolve to meet emerging requirements.

---

## 8. Approval

| Person      | Role           | Approval Date |
| ----------- | -------------- | ------------- |
| Kevin Cogan | Author / ET IC |               |
| PM          |                |               |
| Architect   |                |               |
