# Providers Overview

The goal of Llama Stack is to build an ecosystem where users can easily swap out different implementations for the same API. Examples for these include:
- LLM inference providers (e.g., Meta Reference, Ollama, Fireworks, Together, AWS Bedrock, Groq, Cerebras, SambaNova, vLLM, OpenAI, Anthropic, Gemini, WatsonX, etc.),
- Vector databases (e.g., FAISS, SQLite-Vec, ChromaDB, Weaviate, Qdrant, Milvus, PGVector, etc.),
- Safety providers (e.g., Meta's Llama Guard, Prompt Guard, Code Scanner, AWS Bedrock Guardrails, etc.),
- Tool Runtime providers (e.g., RAG Runtime, Brave Search, etc.)

Providers come in two flavors:
- **Remote**: the provider runs as a separate service external to the Llama Stack codebase. Llama Stack contains a small amount of adapter code.
- **Inline**: the provider is fully specified and implemented within the Llama Stack codebase. It may be a simple wrapper around an existing library, or a full fledged implementation within Llama Stack.

Importantly, Llama Stack always strives to provide at least one fully inline provider for each API so you can iterate on a fully featured environment locally.

## Available Providers

Here is a comprehensive list of all available API providers in Llama Stack:

| API Provider Builder    | Environments      | Agents | Inference | VectorIO | Safety | Telemetry | Post Training | Eval | DatasetIO |Tool Runtime| Scoring |
|:----------------------:|:------------------:|:------:|:---------:|:--------:|:------:|:---------:|:-------------:|:----:|:---------:|:----------:|:-------:|
| Meta Reference         | Single Node        |   ✅   |    ✅     |    ✅    |   ✅   |    ✅     |      ✅      |  ✅  |    ✅     |      ✅    |         |
| SambaNova              | Hosted             |        |    ✅     |          |   ✅   |           |              |      |           |             |         |
| Cerebras               | Hosted             |        |    ✅     |          |        |           |              |      |           |             |         |
| Fireworks              | Hosted             |   ✅   |    ✅     |    ✅    |        |           |              |      |           |             |         |
| AWS Bedrock            | Hosted             |        |    ✅     |          |   ✅   |           |              |      |           |             |         |
| Together               | Hosted             |   ✅   |    ✅     |          |   ✅   |           |              |      |           |             |         |
| Groq                   | Hosted             |        |    ✅     |          |        |           |              |      |           |             |         |
| Ollama                 | Single Node        |        |    ✅     |          |        |           |              |      |           |             |         |
| TGI                    | Hosted/Single Node |        |    ✅     |          |        |           |              |      |           |             |         |
| NVIDIA NIM             | Hosted/Single Node |        |    ✅     |          |   ✅   |           |              |      |           |             |         |
| ChromaDB               | Hosted/Single Node |        |           |    ✅    |        |           |              |      |           |             |         |
| PG Vector              | Single Node        |        |           |    ✅    |        |           |              |      |           |             |         |
| PyTorch ExecuTorch     | On-device iOS      |   ✅   |    ✅     |          |        |           |              |      |           |             |         |
| vLLM                   | Single Node        |        |    ✅     |          |        |           |              |      |           |             |         |
| OpenAI                 | Hosted             |        |    ✅     |          |        |           |              |      |           |             |         |
| Anthropic              | Hosted             |        |    ✅     |          |        |           |              |      |           |             |         |
| Gemini                 | Hosted             |        |    ✅     |          |        |           |              |      |           |             |         |
| WatsonX                | Hosted             |        |    ✅     |          |        |           |              |      |           |             |         |
| HuggingFace            | Single Node        |        |           |          |        |           |      ✅      |      |    ✅     |             |         |
| TorchTune              | Single Node        |        |           |          |        |           |      ✅      |      |           |             |         |
| NVIDIA NEMO            | Hosted             |        |    ✅     |    ✅    |        |           |      ✅      |  ✅  |    ✅     |             |         |
| NVIDIA                 | Hosted             |        |           |          |        |           |      ✅      |  ✅  |    ✅     |             |         |
| FAISS                  | Single Node        |        |           |    ✅    |        |           |              |      |           |             |         |
| SQLite-Vec             | Single Node        |        |           |    ✅    |        |           |              |      |           |             |         |
| Qdrant                 | Hosted/Single Node |        |           |    ✅    |        |           |              |      |           |             |         |
| Weaviate               | Hosted             |        |           |    ✅    |        |           |              |      |           |             |         |
| Milvus                 | Hosted/Single Node |        |           |    ✅    |        |           |              |      |           |             |         |
| Prompt Guard           | Single Node        |        |           |          |   ✅   |           |              |      |           |             |         |
| Llama Guard            | Single Node        |        |           |          |   ✅   |           |              |      |           |             |         |
| Code Scanner           | Single Node        |        |           |          |   ✅   |           |              |      |           |             |         |
| Brave Search           | Hosted             |        |           |          |        |           |              |      |           |      ✅     |         |
| Bing Search            | Hosted             |        |           |          |        |           |              |      |           |      ✅     |         |
| RAG Runtime            | Single Node        |        |           |          |        |           |              |      |           |      ✅     |         |
| Model Context Protocol | Hosted             |        |           |          |        |           |              |      |           |      ✅     |         |
| Sentence Transformers  | Single Node        |        |    ✅     |          |        |           |              |      |           |             |         |
| Braintrust             | Single Node        |        |           |          |        |           |              |      |           |             |    ✅   |
| Basic                  | Single Node        |        |           |          |        |           |              |      |           |             |    ✅   |
| LLM-as-Judge           | Single Node        |        |           |          |        |           |              |      |           |             |    ✅   |
| Databricks             | Hosted             |        |    ✅     |          |        |           |              |      |           |             |         |
| RunPod                 | Hosted             |        |    ✅     |          |        |           |              |      |           |             |         |
| Passthrough            | Hosted             |        |    ✅     |          |        |           |              |      |           |             |         |

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
