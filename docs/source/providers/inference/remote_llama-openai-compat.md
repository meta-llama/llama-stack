# remote::llama-openai-compat

## Description

Llama OpenAI-compatible provider for using Llama models with OpenAI API format.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `api_key` | `pydantic.types.SecretStr \| None` | No |  | The Llama API key |
| `openai_compat_api_base` | `<class 'str'>` | No | https://api.llama.com/compat/v1/ | The URL for the Llama API server |

## Sample Configuration

```yaml
openai_compat_api_base: https://api.llama.com/compat/v1/
api_key: ${env.LLAMA_API_KEY}

```

