# inline::milvus

## Description


Please refer to the remote provider documentation.


## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `db_path` | `<class 'str'>` | No | PydanticUndefined |  |
| `kvstore` | `utils.kvstore.config.RedisKVStoreConfig \| utils.kvstore.config.SqliteKVStoreConfig \| utils.kvstore.config.PostgresKVStoreConfig \| utils.kvstore.config.MongoDBKVStoreConfig` | No | sqlite |  |

## Sample Configuration

```yaml
db_path: ${env.MILVUS_DB_PATH:=~/.llama/dummy}/milvus.db
kvstore:
  type: sqlite
  db_path: ${env.SQLITE_STORE_DIR:=~/.llama/dummy}/milvus_registry.db

```

