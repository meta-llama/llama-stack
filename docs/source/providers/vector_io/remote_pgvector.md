# remote::pgvector

## Description


[PGVector](https://github.com/pgvector/pgvector) is a remote vector database provider for Llama Stack. It
allows you to store and query vectors directly in memory.
That means you'll get fast and efficient vector retrieval.

## Features

- Easy to use
- Fully integrated with Llama Stack

## Usage

To use PGVector in your Llama Stack project, follow these steps:

1. Install the necessary dependencies.
2. Configure your Llama Stack project to use Faiss.
3. Start storing and querying vectors.

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
| `password` | `str \| None` | No | mysecretpassword |  |
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

