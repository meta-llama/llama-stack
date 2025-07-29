# remote::cerebras

## Description

Cerebras inference provider for running models on Cerebras Cloud platform.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `base_url` | `<class 'str'>` | No | https://api.cerebras.ai | Base URL for the Cerebras API |
| `api_key` | `pydantic.types.SecretStr \| None` | No |  | Cerebras API Key |

## Sample Configuration

```yaml
base_url: https://api.cerebras.ai
api_key: ${env.CEREBRAS_API_KEY:=}

```

