# remote::anthropic

## Description

Anthropic inference provider for accessing Claude models and Anthropic's AI services.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `api_key` | `str \| None` | No |  | API key for Anthropic models |

## Sample Configuration

```yaml
api_key: ${env.ANTHROPIC_API_KEY:=}

```

