# inline::chromadb

## Description


[Chroma](https://www.trychroma.com/) is an inline and remote vector
database provider for Llama Stack. It allows you to store and query vectors directly within a Chroma database.
That means you're not limited to storing vectors in memory or in a separate service.

## Features
Chroma supports:
- Store embeddings and their metadata
- Vector search
- Full-text search
- Document storage
- Metadata filtering
- Multi-modal retrieval

## Usage

To use Chrome in your Llama Stack project, follow these steps:

1. Install the necessary dependencies.
2. Configure your Llama Stack project to use chroma.
3. Start storing and querying vectors.

## Installation

You can install chroma using pip:

```bash
pip install chromadb
```

## Documentation
See [Chroma's documentation](https://docs.trychroma.com/docs/overview/introduction) for more details about Chroma in general.



## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `db_path` | `<class 'str'>` | No | PydanticUndefined |  |
| `embedding_model` | `str \| None` | No |  | Optional default embedding model for this provider. If not specified, will use system default. |
| `embedding_dimension` | `int \| None` | No |  | Optional embedding dimension override. Only needed for models with variable dimensions (e.g., Matryoshka embeddings). If not specified, will auto-lookup from model registry. |

## Sample Configuration

```yaml
db_path: ${env.CHROMADB_PATH}

```

