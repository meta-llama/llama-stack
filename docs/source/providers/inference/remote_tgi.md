# remote::tgi

## Description

Text Generation Inference (TGI) provider for HuggingFace model serving.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `url` | `<class 'str'>` | No | PydanticUndefined | The URL for the TGI serving endpoint |

## Sample Configuration

```yaml
url: ${env.TGI_URL:=}

```

