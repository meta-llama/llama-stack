# remote::cerebras-openai-compat

## Description

Cerebras OpenAI-compatible provider for using Cerebras models with OpenAI API format.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `api_key` | `str \| None` | No |  | The Cerebras API key |
| `openai_compat_api_base` | `<class 'str'>` | No | https://api.cerebras.ai/v1 | The URL for the Cerebras API server |

## Sample Configuration

```yaml
openai_compat_api_base: https://api.cerebras.ai/v1
api_key: ${env.CEREBRAS_API_KEY}

```

