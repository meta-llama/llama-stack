# remote::qdrant

## Description


Please refer to the inline provider documentation.


## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `location` | `str \| None` | No |  |  |
| `url` | `str \| None` | No |  |  |
| `port` | `int \| None` | No | 6333 |  |
| `grpc_port` | `<class 'int'>` | No | 6334 |  |
| `prefer_grpc` | `<class 'bool'>` | No | False |  |
| `https` | `bool \| None` | No |  |  |
| `api_key` | `str \| None` | No |  |  |
| `prefix` | `str \| None` | No |  |  |
| `timeout` | `int \| None` | No |  |  |
| `host` | `str \| None` | No |  |  |
| `embedding_model` | `str \| None` | No |  | Optional default embedding model for this provider. If not specified, will use system default. |
| `embedding_dimension` | `int \| None` | No |  | Optional embedding dimension override. Only needed for models with variable dimensions (e.g., Matryoshka embeddings). If not specified, will auto-lookup from model registry. |

## Sample Configuration

```yaml
api_key: ${env.QDRANT_API_KEY}

```

