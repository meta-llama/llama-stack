# inline::vllm

## Description

vLLM inference provider for high-performance model serving with PagedAttention and continuous batching.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `tensor_parallel_size` | `<class 'int'>` | No | 1 | Number of tensor parallel replicas (number of GPUs to use). |
| `max_tokens` | `<class 'int'>` | No | 4096 | Maximum number of tokens to generate. |
| `max_model_len` | `<class 'int'>` | No | 4096 | Maximum context length to use during serving. |
| `max_num_seqs` | `<class 'int'>` | No | 4 | Maximum parallel batch size for generation. |
| `enforce_eager` | `<class 'bool'>` | No | False | Whether to use eager mode for inference (otherwise cuda graphs are used). |
| `gpu_memory_utilization` | `<class 'float'>` | No | 0.3 | How much GPU memory will be allocated when this provider has finished loading, including memory that was already allocated before loading. |

## Sample Configuration

```yaml
tensor_parallel_size: ${env.TENSOR_PARALLEL_SIZE:=1}
max_tokens: ${env.MAX_TOKENS:=4096}
max_model_len: ${env.MAX_MODEL_LEN:=4096}
max_num_seqs: ${env.MAX_NUM_SEQS:=4}
enforce_eager: ${env.ENFORCE_EAGER:=False}
gpu_memory_utilization: ${env.GPU_MEMORY_UTILIZATION:=0.3}

```

