# Providers Overview

The goal of Llama Stack is to build an ecosystem where users can easily swap out different implementations for the same API. Examples for these include:
- LLM inference providers (e.g., Ollama, Fireworks, Together, AWS Bedrock, Groq, Cerebras, SambaNova, vLLM, etc.),
- Vector databases (e.g., ChromaDB, Weaviate, Qdrant, Milvus, FAISS, PGVector, SQLite-Vec, etc.),
- Safety providers (e.g., Meta's Llama Guard, AWS Bedrock Guardrails, etc.)

Providers come in two flavors:
- **Remote**: the provider runs as a separate service external to the Llama Stack codebase. Llama Stack contains a small amount of adapter code.
- **Inline**: the provider is fully specified and implemented within the Llama Stack codebase. It may be a simple wrapper around an existing library, or a full fledged implementation within Llama Stack.

Importantly, Llama Stack always strives to provide at least one fully inline provider for each API so you can iterate on a fully featured environment locally.

## External Providers

Llama Stack supports external providers that live outside of the main codebase. This allows you to create and maintain your own providers independently. See the [External Providers Guide](external) for details.

## Agents
Run multi-step agentic workflows with LLMs with tool usage, memory (RAG), etc.

## DatasetIO
Interfaces with datasets and data loaders.

## Eval
Generates outputs (via Inference or Agents) and perform scoring.

## Inference
Runs inference with an LLM.

## Post Training
Fine-tunes a model.

#### Post Training Providers
The following providers are available for Post Training:

```{toctree}
:maxdepth: 1

external
post_training/huggingface
post_training/torchtune
post_training/nvidia_nemo
```

## Safety
Applies safety policies to the output at a Systems (not only model) level.

## Scoring
Evaluates the outputs of the system.

## Telemetry
Collects telemetry data from the system.

## Tool Runtime
Is associated with the ToolGroup resouces.

## Vector IO

Vector IO refers to operations on vector databases, such as adding documents, searching, and deleting documents.
Vector IO plays a crucial role in [Retreival Augmented Generation (RAG)](../..//building_applications/rag), where the vector
io and database are used to store and retrieve documents for retrieval.

#### Vector IO Providers
The following providers (i.e., databases) are available for Vector IO:

```{toctree}
:maxdepth: 1

external
vector_io/faiss
vector_io/sqlite-vec
vector_io/chromadb
vector_io/pgvector
vector_io/qdrant
vector_io/milvus
vector_io/weaviate
```
