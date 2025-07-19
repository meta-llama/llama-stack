# inline::meta-reference

## Description

Meta's reference implementation of a vector database.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `kvstore` | `utils.kvstore.config.RedisKVStoreConfig \| utils.kvstore.config.SqliteKVStoreConfig \| utils.kvstore.config.PostgresKVStoreConfig \| utils.kvstore.config.MongoDBKVStoreConfig` | No | sqlite |  |
| `embedding_model` | `str \| None` | No |  | Optional default embedding model for this provider. If not specified, will use system default. |
| `embedding_dimension` | `int \| None` | No |  | Optional embedding dimension override. Only needed for models with variable dimensions (e.g., Matryoshka embeddings). If not specified, will auto-lookup from model registry. |

## Sample Configuration

```yaml
kvstore:
  type: sqlite
  db_path: ${env.SQLITE_STORE_DIR:=~/.llama/dummy}/faiss_store.db

```

## Deprecation Notice

⚠️ **Warning**: Please use the `inline::faiss` provider instead.

