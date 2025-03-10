# RFC: Llama-Stack Reranking for RAG Workflows

**Status:** Draft  
**Author:** Kevin Cogan  
**Start Date:** 2025-02-24

## Table of Contents

1. [Abstract](#abstract)  
2. [Introduction](#introduction)  
3. [Problem Statement](#problem-statement)  
4. [Proposed Reranking Solution](#proposed-reranking-solution)  
   4.1. [Extended API Endpoints](#extended-api-endpoints)  
   &nbsp;&nbsp;&nbsp;&nbsp;4.1.1. [Enhanced Query Endpoint](#enhanced-query-endpoint)  
   &nbsp;&nbsp;&nbsp;&nbsp;4.1.2. [Dedicated Rerank Endpoint](#dedicated-rerank-endpoint)  
   4.2. [Data Models and Schemas](#data-models-and-schemas)  
   4.3. [Implementation in the RAG Runtime](#implementation-in-the-rag-runtime)  
   &nbsp;&nbsp;&nbsp;&nbsp;4.3.1. [Inline Reranking Integration](#inline-reranking-integration)  
   &nbsp;&nbsp;&nbsp;&nbsp;4.3.2. [Reranker Service](#reranker-service)  
   4.4. [Example Configuration and Request](#example-configuration-and-request)  
5. [API Design Overview](#api-design-overview)  
6. [Considerations and Tradeoffs](#considerations-and-tradeoffs)  
7. [Conclusion](#conclusion)  
8. [Approvals](#approvals)  


## Abstract
This RFC proposes an enhancement to the Llama-Stack Retrieval-Augmented Generation (RAG) system through a configurable reranking component. Many enterprise users rely on legacy keyword search systems that already have significant investments in content synchronization and indexing. In these cases, re-ranking can improve accuracy by refining search results without requiring a full transition to a vector-based retrieval system. By incorporating an additional scoring step—using either a remote inference API or a self-hosted model—the system enhances document retrieval, providing more precise context for downstream tasks. Users have the flexibility to enable or disable reranking and to select a reranker from remote providers (e.g., LlamaRank, Voyage AI, Cohere) or self-hosted models (e.g., sentence-transformers, LLM-based inference). Additionally, telemetry updates are integrated to capture and report reranking metrics for enhanced observability and performance tuning.
By incorporating an additional scoring step—using either a remote inference API or a self-hosted model—the system enhances document retrieval, providing more precise context for downstream tasks. Users have the flexibility to enable or disable reranking and to select a reranker from remote providers (e.g., LlamaRank, Voyage AI, Cohere) or self-hosted models (e.g., sentence-transformers, LLM-based inference). Additionally, telemetry updates are integrated to capture and report reranking metrics for enhanced observability and performance tuning.  

## Introduction
Current RAG implementations use embedding-based similarity search to retrieve document candidates; however, the preliminary ordering can be suboptimal for ambiguous or complex queries. For enterprise users who rely on keyword-based search systems, re-ranking can be especially impactful, as it enhances accuracy without requiring a full migration to vector search. This document outlines an approach that provides both API-based reranking and inline reranking, ensuring seamless integration with existing retrieval systems while emphasizing configurability, ease of implementation, and robust telemetry reporting.

## Problem Statement
Existing RAG systems efficiently index and retrieve document chunks from vector stores, but they often lack a mechanism to refine initial results. This can lead to suboptimal context for LLMs and hinder overall performance. The case for re-ranking is especially strong for enterprise users relying on legacy keyword search systems, where significant investments have already been made in content synchronization and indexing. In these environments, re-ranking can substantially improve accuracy by refining outputs from established search infrastructure. While new vector stores using state-of-the-art dense models also benefit from re-ranking, the improvements tend to be less pronounced and may not justify the additional complexity and latency. Moreover, different operational needs mean that some users prefer a managed API solution, while others require inline control for low latency or data privacy.

## Proposed Reranking Solution
![My Image](https://drive.google.com/uc?id=115BSpFE3UBmEk7ven5Jq4H7EZd6dMD6U)

## 4.1. Extended API Endpoints

### 4.1.1. Query Endpoint

The `/tool-runtime/rag-tool/query` endpoint will be updated to accept three additional parameters:

- `rerank` (boolean): Enables the reranking process when set to `true`.
- `reranker_model_id` (string): Specifies the reranking provider or model (e.g., `"sentence-transformers"` or self-hosted models).
- `rerank_config` (`Optional[RAGRerankConfig]`): Configures additional options for the reranking process (e.g., `is_external_reranker`, `api_url`, `api_key`).

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
        rerank: bool = False,
        reranker_model_id: str = "my_model_id",
        rerank_config: Optional[RAGRerankConfig] = None
    ) -> RAGQueryResult:
        ...
```
> <sub>**Note:** Note: When rerank is enabled, the service will invoke the reranking process using the specified reranker_model_id and additional options defined in rerank_config.


