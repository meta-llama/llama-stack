# inline::milvus

## Description

[Milvus](https://milvus.io/) is an inline vector database provider for Llama Stack that provides high-performance vector similarity search and analytics. It offers a robust, scalable solution for managing vector embeddings with advanced indexing and search capabilities.

## Features

- **High Performance**: Optimized for large-scale vector operations
- **Multiple Index Types**: Support for various vector indexing algorithms (IVF, HNSW, etc.)
- **Scalability**: Designed to handle millions to billions of vectors
- **FileResponse Support**: Full integration with Llama Stack Files API for document processing
- **OpenAI Vector Store Compatibility**: Support for attaching files to vector stores with automatic chunking
- **Advanced Search**: Support for vector similarity search with configurable parameters

## OpenAI-Compatible File Operations

Milvus supports OpenAI-compatible file operations, allowing you to:

- **Attach files to vector stores**: Upload documents and automatically process them into searchable chunks
- **File management**: List, retrieve, and manage files within vector stores
- **Automatic chunking**: Files are automatically split into optimal chunks for vector search
- **Metadata preservation**: File attributes and metadata are preserved during processing

## Search Modes

**Supported:**
- **Vector Search**: Performs vector similarity search using embeddings
- **Filtered Search**: Combine vector search with metadata filtering
- **Range Search**: Search within specific distance thresholds

## Usage

To use Milvus in your Llama Stack project:

1. Install the necessary dependencies
2. Configure your Llama Stack project to use Milvus
3. Start storing and querying vectors

## Installation

You can install Milvus using pip:

```bash
pip install pymilvus
```

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `db_path` | `<class 'str'>` | No |  | Path to the Milvus database file |
| `kvstore` | `utils.kvstore.config.RedisKVStoreConfig \| utils.kvstore.config.SqliteKVStoreConfig \| utils.kvstore.config.PostgresKVStoreConfig \| utils.kvstore.config.MongoDBKVStoreConfig` | No | sqlite | Config for KV store backend (SQLite only for now) |
| `consistency_level` | `<class 'str'>` | No | Strong | The consistency level of the Milvus server |

## Sample Configuration

```yaml
db_path: ${env.MILVUS_DB_PATH:=~/.llama/dummy}/milvus.db
kvstore:
  type: sqlite
  db_path: ${env.SQLITE_STORE_DIR:=~/.llama/dummy}/milvus_registry.db
consistency_level: Strong
```

## Documentation

For more information about Milvus, see the [official Milvus documentation](https://milvus.io/docs).

