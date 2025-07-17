# API Providers Overview

The goal of Llama Stack is to build an ecosystem where users can easily swap out different implementations for the same API. Examples for these include:
- LLM inference providers (e.g., Meta Reference, Ollama, Fireworks, Together, AWS Bedrock, Groq, Cerebras, SambaNova, vLLM, OpenAI, Anthropic, Gemini, WatsonX, etc.),
- Vector databases (e.g., FAISS, SQLite-Vec, ChromaDB, Weaviate, Qdrant, Milvus, PGVector, etc.),
- Safety providers (e.g., Meta's Llama Guard, Prompt Guard, Code Scanner, AWS Bedrock Guardrails, etc.),
- Tool Runtime providers (e.g., RAG Runtime, Brave Search, etc.)

Providers come in two flavors:
- **Remote**: the provider runs as a separate service external to the Llama Stack codebase. Llama Stack contains a small amount of adapter code.
- **Inline**: the provider is fully specified and implemented within the Llama Stack codebase. It may be a simple wrapper around an existing library, or a full fledged implementation within Llama Stack.

Importantly, Llama Stack always strives to provide at least one fully inline provider for each API so you can iterate on a fully featured environment locally.

## External Providers
Llama Stack supports external providers that live outside of the main codebase. This allows you to create and maintain your own providers independently.

```{toctree}
:maxdepth: 1

external.md
```

```{include} openai.md
:start-after: ## OpenAI API Compatibility
```

## Inference
Runs inference with an LLM.

```{toctree}
:maxdepth: 1

inference/index
```

## Agents
Run multi-step agentic workflows with LLMs with tool usage, memory (RAG), etc.

```{toctree}
:maxdepth: 1

agents/index
```

## DatasetIO
Interfaces with datasets and data loaders.

```{toctree}
:maxdepth: 1

datasetio/index
```

## Safety
Applies safety policies to the output at a Systems (not only model) level.

```{toctree}
:maxdepth: 1

safety/index
```

## Telemetry
Collects telemetry data from the system.

```{toctree}
:maxdepth: 1

telemetry/index
```

## Vector IO

Vector IO refers to operations on vector databases, such as adding documents, searching, and deleting documents.
Vector IO plays a crucial role in [Retreival Augmented Generation (RAG)](../..//building_applications/rag), where the vector
io and database are used to store and retrieve documents for retrieval.

```{toctree}
:maxdepth: 1

vector_io/index
```

## Tool Runtime
Is associated with the ToolGroup resources.

```{toctree}
:maxdepth: 1

tool_runtime/index
```