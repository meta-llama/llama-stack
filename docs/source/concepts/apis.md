## APIs

A Llama Stack API is described as a collection of REST endpoints. We currently support the following APIs:

- **Inference**: run inference with a LLM
- **Safety**: apply safety policies to the output at a Systems (not only model) level
- **Agents**: run multi-step agentic workflows with LLMs with tool usage, memory (RAG), etc.
- **DatasetIO**: interface with datasets and data loaders
- **Scoring**: evaluate outputs of the system
- **Eval**: generate outputs (via Inference or Agents) and perform scoring
- **VectorIO**: perform operations on vector stores, such as adding documents, searching, and deleting documents
- **Files**: manage file uploads, storage, and retrieval with OpenAI-compatible API endpoints
- **Telemetry**: collect telemetry data from the system
- **Post Training**: fine-tune a model
- **Tool Runtime**: interact with various tools and protocols
- **Responses**: generate responses from an LLM using this OpenAI compatible API.

We are working on adding a few more APIs to complete the application lifecycle. These will include:
- **Batch Inference**: run inference on a dataset of inputs
- **Batch Agents**: run agents on a dataset of inputs
- **Synthetic Data Generation**: generate synthetic data for model development
- **Batches**: OpenAI-compatible batch management for inference

## OpenAI-Compatible File Operations and Vector Store Integration

The Files API and Vector Store APIs work together through OpenAI-compatible file operations, enabling automatic document processing and search. This integration implements the [OpenAI Vector Store Files API specification](https://platform.openai.com/docs/api-reference/vector-stores-files) and allows you to:

- Upload documents through the Files API
- Automatically process and chunk documents into searchable vectors
- Store processed content in vector databases (FAISS, SQLite-vec, Milvus, etc.)
- Search through documents using natural language queries

For detailed information about this integration, see [OpenAI-Compatible File Operations and Vector Store Integration](openai_file_operations_vector_stores.md).
