# remote::runpod

## Description

RunPod inference provider for running models on RunPod's cloud GPU platform.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `url` | `str \| None` | No |  | The URL for the Runpod model serving endpoint |
| `api_token` | `pydantic.types.SecretStr \| None` | No |  | The API token |

## Sample Configuration

```yaml
url: ${env.RUNPOD_URL:=}
api_token: ${env.RUNPOD_API_TOKEN}

```

