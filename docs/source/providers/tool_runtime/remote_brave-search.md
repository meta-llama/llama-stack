# remote::brave-search

## Description

Brave Search tool for web search capabilities with privacy-focused results.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `api_key` | `pydantic.types.SecretStr \| None` | No |  | The Brave Search API Key |
| `max_results` | `<class 'int'>` | No | 3 | The maximum number of results to return |

## Sample Configuration

```yaml
api_key: ${env.BRAVE_SEARCH_API_KEY:=}
max_results: 3

```

