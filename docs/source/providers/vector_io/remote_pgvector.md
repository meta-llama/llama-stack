# remote::pgvector

## Description


[PGVector](https://github.com/pgvector/pgvector) is a remote vector database provider for Llama Stack. It
allows you to store and query vectors directly in memory.
That means you'll get fast and efficient vector retrieval.

## Features

- Easy to use
- Fully integrated with Llama Stack

There are three implementations of search for PGVectoIndex available:

1. Vector Search:
- How it works:
  - Uses PostgreSQL's vector extension (pgvector) to perform similarity search
  - Compares query embeddings against stored embeddings using Cosine distance or other distance metrics
  - Eg. SQL query: SELECT document, embedding <=> %s::vector AS distance FROM table ORDER BY distance

-Characteristics:
  - Semantic understanding - finds documents similar in meaning even if they don't share keywords
  - Works with high-dimensional vector embeddings (typically 768, 1024, or higher dimensions)
  - Best for: Finding conceptually related content, handling synonyms, cross-language search

2. Keyword Search
- How it works:
  - Uses PostgreSQL's full-text search capabilities with tsvector and ts_rank
  - Converts text to searchable tokens using to_tsvector('english', text). Default language is English.
  - Eg. SQL query: SELECT document, ts_rank(tokenized_content, plainto_tsquery('english', %s)) AS score

- Characteristics:
  - Lexical matching - finds exact keyword matches and variations
  - Uses GIN (Generalized Inverted Index) for fast text search performance
  - Scoring: Uses PostgreSQL's ts_rank function for relevance scoring
  - Best for: Exact term matching, proper names, technical terms, Boolean-style queries

3. Hybrid Search
- How it works:
  - Combines both vector and keyword search results
  - Runs both searches independently, then merges results using configurable reranking

- Two reranking strategies available:
    - Reciprocal Rank Fusion (RRF) - (default: 60.0)
    - Weighted Average - (default: 0.5)

- Characteristics:
  - Best of both worlds: semantic understanding + exact matching
  - Documents appearing in both searches get boosted scores
  - Configurable balance between semantic and lexical matching
  - Best for: General-purpose search where you want both precision and recall

4. Database Schema
The PGVector implementation stores data optimized for all three search types:
CREATE TABLE vector_store_xxx (
    id TEXT PRIMARY KEY,
    document JSONB,                    -- Original document
    embedding vector(dimension),        -- For vector search
    content_text TEXT,                 -- Raw text content
    tokenized_content TSVECTOR          -- For keyword search
);

-- Indexes for performance
CREATE INDEX content_gin_idx ON table USING GIN(tokenized_content);  -- Keyword search
-- Vector index created automatically by pgvector

## Usage

To use PGVector in your Llama Stack project, follow these steps:

1. Install the necessary dependencies.
2. Configure your Llama Stack project to use pgvector. (e.g. remote::pgvector).
3. Start storing and querying vectors.

## This is an example how you can set up your environment for using PGVector

1. Export env vars:
```bash
export ENABLE_PGVECTOR=true
export PGVECTOR_HOST=localhost
export PGVECTOR_PORT=5432
export PGVECTOR_DB=llamastack
export PGVECTOR_USER=llamastack
export PGVECTOR_PASSWORD=llamastack
```

2. Create DB:
```bash
psql -h localhost -U postgres -c "CREATE ROLE llamastack LOGIN PASSWORD 'llamastack';"
psql -h localhost -U postgres -c "CREATE DATABASE llamastack OWNER llamastack;"
psql -h localhost -U llamastack -d llamastack -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Installation

You can install PGVector using docker:

```bash
docker pull pgvector/pgvector:pg17
```
## Documentation
See [PGVector's documentation](https://github.com/pgvector/pgvector) for more details about PGVector in general.


## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `host` | `str \| None` | No | localhost |  |
| `port` | `int \| None` | No | 5432 |  |
| `db` | `str \| None` | No | postgres |  |
| `user` | `str \| None` | No | postgres |  |
| `password` | `pydantic.types.SecretStr \| None` | No | mysecretpassword |  |
| `kvstore` | `utils.kvstore.config.RedisKVStoreConfig \| utils.kvstore.config.SqliteKVStoreConfig \| utils.kvstore.config.PostgresKVStoreConfig \| utils.kvstore.config.MongoDBKVStoreConfig, annotation=NoneType, required=False, default='sqlite', discriminator='type'` | No |  | Config for KV store backend (SQLite only for now) |

## Sample Configuration

```yaml
host: ${env.PGVECTOR_HOST:=localhost}
port: ${env.PGVECTOR_PORT:=5432}
db: ${env.PGVECTOR_DB}
user: ${env.PGVECTOR_USER}
password: ${env.PGVECTOR_PASSWORD}
kvstore:
  type: sqlite
  db_path: ${env.SQLITE_STORE_DIR:=~/.llama/dummy}/pgvector_registry.db

```

