# inline::sqlite_vec

## Description


Please refer to the sqlite-vec provider documentation.


## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `db_path` | `<class 'str'>` | No | PydanticUndefined |  |

## Sample Configuration

```yaml
db_path: ${env.SQLITE_STORE_DIR:=~/.llama/dummy}/sqlite_vec.db

```

## Deprecation Notice

⚠️ **Warning**: Please use the `inline::sqlite-vec` provider (notice the hyphen instead of underscore) instead.

