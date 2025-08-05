# inline::huggingface

## Description

HuggingFace-based post-training provider for fine-tuning models using the HuggingFace ecosystem.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `device` | `<class 'str'>` | No | cuda |  |
| `distributed_backend` | `Literal['fsdp', 'deepspeed'` | No |  |  |
| `checkpoint_format` | `Literal['full_state', 'huggingface'` | No | huggingface |  |
| `chat_template` | `<class 'str'>` | No | <|user|>
{input}
<|assistant|>
{output} |  |
| `model_specific_config` | `<class 'dict'>` | No | {'trust_remote_code': True, 'attn_implementation': 'sdpa'} |  |
| `max_seq_length` | `<class 'int'>` | No | 2048 |  |
| `gradient_checkpointing` | `<class 'bool'>` | No | False |  |
| `save_total_limit` | `<class 'int'>` | No | 3 |  |
| `logging_steps` | `<class 'int'>` | No | 10 |  |
| `warmup_ratio` | `<class 'float'>` | No | 0.1 |  |
| `weight_decay` | `<class 'float'>` | No | 0.01 |  |
| `dataloader_num_workers` | `<class 'int'>` | No | 4 |  |
| `dataloader_pin_memory` | `<class 'bool'>` | No | True |  |
| `dpo_beta` | `<class 'float'>` | No | 0.1 |  |
| `use_reference_model` | `<class 'bool'>` | No | True |  |
| `dpo_loss_type` | `Literal['sigmoid', 'hinge', 'ipo', 'kto_pair'` | No | sigmoid |  |
| `dpo_output_dir` | `<class 'str'>` | No | /var/folders/mj/t_stv1ys7637vqf2_b4yf67m0000gn/T/dpo_output_q24jwy0r |  |

## Sample Configuration

```yaml
checkpoint_format: huggingface
distributed_backend: null
device: cpu

```

