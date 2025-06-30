# remote::groq-openai-compat

## Description

Groq OpenAI-compatible provider for using Groq models with OpenAI API format.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `api_key` | `str \| None` | No |  | The Groq API key |
| `openai_compat_api_base` | `<class 'str'>` | No | https://api.groq.com/openai/v1 | The URL for the Groq API server |

## Sample Configuration

```yaml
openai_compat_api_base: https://api.groq.com/openai/v1
api_key: ${env.GROQ_API_KEY}

```

