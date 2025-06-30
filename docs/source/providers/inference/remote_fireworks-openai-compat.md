# remote::fireworks-openai-compat

## Description

Fireworks AI OpenAI-compatible provider for using Fireworks models with OpenAI API format.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `api_key` | `str \| None` | No |  | The Fireworks API key |
| `openai_compat_api_base` | `<class 'str'>` | No | https://api.fireworks.ai/inference/v1 | The URL for the Fireworks API server |

## Sample Configuration

```yaml
openai_compat_api_base: https://api.fireworks.ai/inference/v1
api_key: ${env.FIREWORKS_API_KEY}

```

