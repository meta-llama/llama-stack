# remote::llamacpp

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `api_key` | `str \| None` | No |  | The llama.cpp server API key (optional for local servers) |
| `openai_compat_api_base` | `<class 'str'>` | No | http://localhost:8080/v1 | The URL for the llama.cpp server with OpenAI-compatible API |

## Sample Configuration

```yaml
openai_compat_api_base: ${env.LLAMACPP_URL:http://localhost:8080}/v1
api_key: ${env.LLAMACPP_API_KEY:=}

```

