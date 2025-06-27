# remote::openai

## Description

OpenAI inference provider for accessing GPT models and other OpenAI services.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `api_key` | `str \| None` | No |  | API key for OpenAI models |

## Sample Configuration

```yaml
api_key: ${env.OPENAI_API_KEY}

```

