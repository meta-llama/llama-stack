# inline::meta-reference

## Description

Meta's reference implementation of a vector database.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `kvstore` | `utils.kvstore.config.RedisKVStoreConfig \| utils.kvstore.config.SqliteKVStoreConfig \| utils.kvstore.config.PostgresKVStoreConfig \| utils.kvstore.config.MongoDBKVStoreConfig` | No | sqlite |  |
| `embedding` | `utils.vector_io.embedding_config.EmbeddingConfig \| None` | No |  | Default embedding configuration for this provider. When specified, vector databases created with this provider will use these embedding settings as defaults. |

## Sample Configuration

```yaml
kvstore:
  type: sqlite
  db_path: ${env.SQLITE_STORE_DIR:=~/.llama/dummy}/faiss_store.db

```

## Deprecation Notice

⚠️ **Warning**: Please use the `inline::faiss` provider instead.

