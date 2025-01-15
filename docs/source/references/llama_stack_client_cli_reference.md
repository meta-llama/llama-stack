# llama (client-side) CLI Reference

The `llama-stack-client` CLI allows you to query information about the distribution.

## Basic Commands

### `llama-stack-client`
```bash
$ llama-stack-client -h

usage: llama-stack-client [-h] {models,memory_banks,shields} ...

Welcome to the LlamaStackClient CLI

options:
  -h, --help            show this help message and exit

subcommands:
  {models,memory_banks,shields}
```

### `llama-stack-client configure`
```bash
$ llama-stack-client configure
> Enter the host name of the Llama Stack distribution server: localhost
> Enter the port number of the Llama Stack distribution server: 5000
Done! You can now use the Llama Stack Client CLI with endpoint http://localhost:5000
```

### `llama-stack-client providers list`
```bash
$ llama-stack-client providers list
```
```
+-----------+----------------+-----------------+
| API       | Provider ID    | Provider Type   |
+===========+================+=================+
| scoring   | meta0          | meta-reference  |
+-----------+----------------+-----------------+
| datasetio | meta0          | meta-reference  |
+-----------+----------------+-----------------+
| inference | tgi0           | remote::tgi     |
+-----------+----------------+-----------------+
| memory    | meta-reference | meta-reference  |
+-----------+----------------+-----------------+
| agents    | meta-reference | meta-reference  |
+-----------+----------------+-----------------+
| telemetry | meta-reference | meta-reference  |
+-----------+----------------+-----------------+
| safety    | meta-reference | meta-reference  |
+-----------+----------------+-----------------+
```

## Model Management

### `llama-stack-client models list`
```bash
$ llama-stack-client models list
```
```
+----------------------+----------------------+---------------+----------------------------------------------------------+
| identifier           | llama_model          | provider_id   | metadata                                                 |
+======================+======================+===============+==========================================================+
| Llama3.1-8B-Instruct | Llama3.1-8B-Instruct | tgi0          | {'huggingface_repo': 'meta-llama/Llama-3.1-8B-Instruct'} |
+----------------------+----------------------+---------------+----------------------------------------------------------+
```

### `llama-stack-client models get`
```bash
$ llama-stack-client models get Llama3.1-8B-Instruct
```

```
+----------------------+----------------------+----------------------------------------------------------+---------------+
| identifier           | llama_model          | metadata                                                 | provider_id   |
+======================+======================+==========================================================+===============+
| Llama3.1-8B-Instruct | Llama3.1-8B-Instruct | {'huggingface_repo': 'meta-llama/Llama-3.1-8B-Instruct'} | tgi0          |
+----------------------+----------------------+----------------------------------------------------------+---------------+
```


```bash
$ llama-stack-client models get Random-Model

Model RandomModel is not found at distribution endpoint host:port. Please ensure endpoint is serving specified model.
```

### `llama-stack-client models register`

```bash
$ llama-stack-client models register <model_id> [--provider-id <provider_id>] [--provider-model-id <provider_model_id>] [--metadata <metadata>]
```

### `llama-stack-client models update`

```bash
$ llama-stack-client models update <model_id> [--provider-id <provider_id>] [--provider-model-id <provider_model_id>] [--metadata <metadata>]
```

### `llama-stack-client models delete`

```bash
$ llama-stack-client models delete <model_id>
```

## Memory Bank Management

### `llama-stack-client memory_banks list`
```bash
$ llama-stack-client memory_banks list
```
```
+--------------+----------------+--------+-------------------+------------------------+--------------------------+
| identifier   | provider_id    | type   | embedding_model   |   chunk_size_in_tokens |   overlap_size_in_tokens |
+==============+================+========+===================+========================+==========================+
| test_bank    | meta-reference | vector | all-MiniLM-L6-v2  |                    512 |                       64 |
+--------------+----------------+--------+-------------------+------------------------+--------------------------+
```

### `llama-stack-client memory_banks register`
```bash
$ llama-stack-client memory_banks register <memory-bank-id> --type <type> [--provider-id <provider-id>] [--provider-memory-bank-id <provider-memory-bank-id>] [--chunk-size <chunk-size>] [--embedding-model <embedding-model>] [--overlap-size <overlap-size>]
```

Options:
- `--type`: Required. Type of memory bank. Choices: "vector", "keyvalue", "keyword", "graph"
- `--provider-id`: Optional. Provider ID for the memory bank
- `--provider-memory-bank-id`: Optional. Provider's memory bank ID
- `--chunk-size`: Optional. Chunk size in tokens (for vector type). Default: 512
- `--embedding-model`: Optional. Embedding model (for vector type). Default: "all-MiniLM-L6-v2"
- `--overlap-size`: Optional. Overlap size in tokens (for vector type). Default: 64

### `llama-stack-client memory_banks unregister`
```bash
$ llama-stack-client memory_banks unregister <memory-bank-id>
```

## Shield Management
### `llama-stack-client shields list`
```bash
$ llama-stack-client shields list
```

```
+--------------+----------+----------------+-------------+
| identifier   | params   | provider_id    | type        |
+==============+==========+================+=============+
| llama_guard  | {}       | meta-reference | llama_guard |
+--------------+----------+----------------+-------------+
```

### `llama-stack-client shields register`
```bash
$ llama-stack-client shields register --shield-id <shield-id> [--provider-id <provider-id>] [--provider-shield-id <provider-shield-id>] [--params <params>]
```

Options:
- `--shield-id`: Required. ID of the shield
- `--provider-id`: Optional. Provider ID for the shield
- `--provider-shield-id`: Optional. Provider's shield ID
- `--params`: Optional. JSON configuration parameters for the shield

## Eval Task Management

### `llama-stack-client eval_tasks list`
```bash
$ llama-stack-client eval_tasks list
```

### `llama-stack-client eval_tasks register`
```bash
$ llama-stack-client eval_tasks register --eval-task-id <eval-task-id> --dataset-id <dataset-id> --scoring-functions <function1> [<function2> ...] [--provider-id <provider-id>] [--provider-eval-task-id <provider-eval-task-id>] [--metadata <metadata>]
```

Options:
- `--eval-task-id`: Required. ID of the eval task
- `--dataset-id`: Required. ID of the dataset to evaluate
- `--scoring-functions`: Required. One or more scoring functions to use for evaluation
- `--provider-id`: Optional. Provider ID for the eval task
- `--provider-eval-task-id`: Optional. Provider's eval task ID
- `--metadata`: Optional. Metadata for the eval task in JSON format

## Eval execution
### `llama-stack-client eval run-benchmark`
```bash
$ llama-stack-client eval run-benchmark <eval-task-id1> [<eval-task-id2> ...] --eval-task-config <config-file> --output-dir <output-dir> [--num-examples <num>] [--visualize]
```

Options:
- `--eval-task-config`: Required. Path to the eval task config file in JSON format
- `--output-dir`: Required. Path to the directory where evaluation results will be saved
- `--num-examples`: Optional. Number of examples to evaluate (useful for debugging)
- `--visualize`: Optional flag. If set, visualizes evaluation results after completion

Example eval_task_config.json:
```json
{
    "type": "benchmark",
    "eval_candidate": {
        "type": "model",
        "model": "Llama3.1-405B-Instruct",
        "sampling_params": {
            "strategy": {
              "type": "greedy"
            },
            "max_tokens": 0,
            "repetition_penalty": 1.0
        }
    }
}
```

### `llama-stack-client eval run-scoring`
```bash
$ llama-stack-client eval run-scoring <eval-task-id> --eval-task-config <config-file> --output-dir <output-dir> [--num-examples <num>] [--visualize]
```

Options:
- `--eval-task-config`: Required. Path to the eval task config file in JSON format
- `--output-dir`: Required. Path to the directory where scoring results will be saved
- `--num-examples`: Optional. Number of examples to evaluate (useful for debugging)
- `--visualize`: Optional flag. If set, visualizes scoring results after completion
