# remote::together-openai-compat

## Description

Together AI OpenAI-compatible provider for using Together models with OpenAI API format.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `api_key` | `str \| None` | No |  | The Together API key |
| `openai_compat_api_base` | `<class 'str'>` | No | https://api.together.xyz/v1 | The URL for the Together API server |

## Sample Configuration

```yaml
openai_compat_api_base: https://api.together.xyz/v1
api_key: ${env.TOGETHER_API_KEY}

```

