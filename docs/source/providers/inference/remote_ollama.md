# remote::ollama

## Description

Ollama inference provider for running local models through the Ollama runtime.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `url` | `<class 'str'>` | No | http://localhost:11434 |  |

## Sample Configuration

```yaml
url: ${env.OLLAMA_URL:=http://localhost:11434}

```

