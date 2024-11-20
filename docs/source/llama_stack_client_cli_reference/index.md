# llama-stack-client CLI Reference

You may use the `llama-stack-client` to query information about the distribution.

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

## Provider Commands

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

## Evaluation Tasks

### `llama-stack-client eval_tasks list`
```bash
$ llama-stack-client eval run_benchmark <task_id1> <task_id2> --num-examples 10 --output-dir ./ --eval-task-config ~/eval_task_config.json
```

where `eval_task_config.json` is the path to the eval task config file in JSON format. An example eval_task_config
```
$ cat ~/eval_task_config.json
{
    "type": "benchmark",
    "eval_candidate": {
        "type": "model",
        "model": "Llama3.1-405B-Instruct",
        "sampling_params": {
            "strategy": "greedy",
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 0,
            "max_tokens": 0,
            "repetition_penalty": 1.0
        }
    }
}
```
