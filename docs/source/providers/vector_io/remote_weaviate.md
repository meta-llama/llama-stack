# remote::weaviate

## Description


[Weaviate](https://weaviate.io/) is a vector database provider for Llama Stack.
It allows you to store and query vectors directly within a Weaviate database.
That means you're not limited to storing vectors in memory or in a separate service.

## Features
Weaviate supports:
- Store embeddings and their metadata
- Vector search
- Full-text search
- Hybrid search
- Document storage
- Metadata filtering
- Multi-modal retrieval

## Usage

To use Weaviate in your Llama Stack project, follow these steps:

1. Install the necessary dependencies.
2. Configure your Llama Stack project to use chroma.
3. Start storing and querying vectors.

## Installation

To install Weaviate see the [Weaviate quickstart documentation](https://weaviate.io/developers/weaviate/quickstart).

## Documentation
See [Weaviate's documentation](https://weaviate.io/developers/weaviate) for more details about Weaviate in general.


## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `embedding_model` | `str \| None` | No |  | Optional default embedding model for this provider. If not specified, will use system default. |
| `embedding_dimension` | `int \| None` | No |  | Optional embedding dimension override. Only needed for models with variable dimensions (e.g., Matryoshka embeddings). If not specified, will auto-lookup from model registry. |

## Sample Configuration

```yaml
kvstore:
  type: sqlite
  db_path: ${env.SQLITE_STORE_DIR:=~/.llama/dummy}/weaviate_registry.db

```

