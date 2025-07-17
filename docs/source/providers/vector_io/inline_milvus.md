# inline::milvus

## Description


Please refer to the remote provider documentation.


## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `db_path` | `<class 'str'>` | No | PydanticUndefined |  |
| `kvstore` | `utils.kvstore.config.RedisKVStoreConfig \| utils.kvstore.config.SqliteKVStoreConfig \| utils.kvstore.config.PostgresKVStoreConfig \| utils.kvstore.config.MongoDBKVStoreConfig` | No | sqlite | Config for KV store backend (SQLite only for now) |
| `consistency_level` | `<class 'str'>` | No | Strong | The consistency level of the Milvus server |
| `embedding_model` | `str \| None` | No |  | Optional default embedding model for this provider. If not specified, will use system default. |
| `embedding_dimension` | `int \| None` | No |  | Optional embedding dimension override. Only needed for models with variable dimensions (e.g., Matryoshka embeddings). If not specified, will auto-lookup from model registry. |

## Sample Configuration

```yaml
db_path: ${env.MILVUS_DB_PATH:=~/.llama/dummy}/milvus.db
kvstore:
  type: sqlite
  db_path: ${env.SQLITE_STORE_DIR:=~/.llama/dummy}/milvus_registry.db

```

