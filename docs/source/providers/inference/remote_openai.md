# remote::openai

## Description

OpenAI inference provider for accessing GPT models and other OpenAI services.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `api_key` | `str \| None` | No |  | API key for OpenAI models |
| `base_url` | `<class 'str'>` | No | https://api.openai.com/v1 | Base URL for OpenAI API |

## Sample Configuration

```yaml
api_key: ${env.OPENAI_API_KEY:=}
base_url: ${env.OPENAI_BASE_URL:=https://api.openai.com/v1}

```

