# Providers Overview

The goal of Llama Stack is to build an ecosystem where users can easily swap out different implementations for the same API. Examples for these include:
- LLM inference providers (e.g., Fireworks, Together, AWS Bedrock, Groq, Cerebras, SambaNova, etc.),
- Vector databases (e.g., ChromaDB, Weaviate, Qdrant, FAISS, PGVector, etc.),
- Safety providers (e.g., Meta's Llama Guard, AWS Bedrock Guardrails, etc.)

Providers come in two flavors:
- **Remote**: the provider runs as a separate service external to the Llama Stack codebase. Llama Stack contains a small amount of adapter code.
- **Inline**: the provider is fully specified and implemented within the Llama Stack codebase. It may be a simple wrapper around an existing library, or a full fledged implementation within Llama Stack.

Importantly, Llama Stack always strives to provide at least one fully "local" provider for each API so you can iterate on a fully featured environment locally.

## Agents

## DatasetIO

## Eval

## Inference

## iOS

## Post Training

## Safety

## Scoring

## Telemetry

## Tool Runtime

## [Vector DBs](vector_db/index.md)

```{toctree}
:maxdepth: 1

vector_db/chromadb
vector_db/sqlite-vec
vector_db/faiss
vector_db/pgvector
vector_db/qdrant
vector_db/weaviate
```
