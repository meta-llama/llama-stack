# remote::tavily-search

## Description

Tavily Search tool for AI-optimized web search with structured results.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `api_key` | `pydantic.types.SecretStr \| None` | No |  | The Tavily Search API Key |
| `max_results` | `<class 'int'>` | No | 3 | The maximum number of results to return |

## Sample Configuration

```yaml
api_key: ${env.TAVILY_SEARCH_API_KEY:=}
max_results: 3

```

