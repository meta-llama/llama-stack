# remote::fireworks

## Description

Fireworks AI inference provider for Llama models and other AI models on the Fireworks platform.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `url` | `<class 'str'>` | No | https://api.fireworks.ai/inference/v1 | The URL for the Fireworks server |
| `api_key` | `pydantic.types.SecretStr \| None` | No |  | The Fireworks.ai API Key |

## Sample Configuration

```yaml
url: https://api.fireworks.ai/inference/v1
api_key: ${env.FIREWORKS_API_KEY}

```

