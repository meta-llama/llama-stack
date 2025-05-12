# llama (client-side) CLI Reference

The `llama-stack-client` CLI allows you to query information about the distribution.

## Basic Commands

### `llama-stack-client`
```bash
llama-stack-client
Usage: llama-stack-client [OPTIONS] COMMAND [ARGS]...

  Welcome to the LlamaStackClient CLI

Options:
  --version        Show the version and exit.
  --endpoint TEXT  Llama Stack distribution endpoint
  --api-key TEXT   Llama Stack distribution API key
  --config TEXT    Path to config file
  --help           Show this message and exit.

Commands:
  configure          Configure Llama Stack Client CLI.
  datasets           Manage datasets.
  eval               Run evaluation tasks.
  eval_tasks         Manage evaluation tasks.
  inference          Inference (chat).
  inspect            Inspect server configuration.
  models             Manage GenAI models.
  post_training      Post-training.
  providers          Manage API providers.
  scoring_functions  Manage scoring functions.
  shields            Manage safety shield services.
  toolgroups         Manage available tool groups.
  vector_dbs         Manage vector databases.
```

### `llama-stack-client configure`
```bash
llama-stack-client configure
> Enter the host name of the Llama Stack distribution server: localhost
> Enter the port number of the Llama Stack distribution server: 8321
Done! You can now use the Llama Stack Client CLI with endpoint http://localhost:8321
```

### `llama-stack-client providers list`
```bash
llama-stack-client providers list
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
llama-stack-client models list
```
```
Available Models

┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ model_type   ┃ identifier                           ┃ provider_resource_id         ┃ metadata  ┃ provider_id ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ llm          │ meta-llama/Llama-3.2-3B-Instruct     │ llama3.2:3b-instruct-fp16    │           │ ollama      │
└──────────────┴──────────────────────────────────────┴──────────────────────────────┴───────────┴─────────────┘

Total models: 1
```

### `llama-stack-client models get`
```bash
llama-stack-client models get Llama3.1-8B-Instruct
```

```
+----------------------+----------------------+----------------------------------------------------------+---------------+
| identifier           | llama_model          | metadata                                                 | provider_id   |
+======================+======================+==========================================================+===============+
| Llama3.1-8B-Instruct | Llama3.1-8B-Instruct | {'huggingface_repo': 'meta-llama/Llama-3.1-8B-Instruct'} | tgi0          |
+----------------------+----------------------+----------------------------------------------------------+---------------+
```


```bash
llama-stack-client models get Random-Model

Model RandomModel is not found at distribution endpoint host:port. Please ensure endpoint is serving specified model.
```

### `llama-stack-client models register`

```bash
llama-stack-client models register <model_id> [--provider-id <provider_id>] [--provider-model-id <provider_model_id>] [--metadata <metadata>]
```

### `llama-stack-client models update`

```bash
llama-stack-client models update <model_id> [--provider-id <provider_id>] [--provider-model-id <provider_model_id>] [--metadata <metadata>]
```

### `llama-stack-client models delete`

```bash
llama-stack-client models delete <model_id>
```

## Vector DB Management

### `llama-stack-client vector_dbs list`
```bash
llama-stack-client vector_dbs list
```
```
+--------------+----------------+---------------------+---------------+------------------------+
| identifier   | provider_id    | provider_resource_id| vector_db_type| params                |
+==============+================+=====================+===============+========================+
| test_bank    | meta-reference | test_bank          | vector        | embedding_model: all-MiniLM-L6-v2
                                                                      embedding_dimension: 384|
+--------------+----------------+---------------------+---------------+------------------------+
```

### `llama-stack-client vector_dbs register`
```bash
llama-stack-client vector_dbs register <vector-db-id> [--provider-id <provider-id>] [--provider-vector-db-id <provider-vector-db-id>] [--embedding-model <embedding-model>] [--embedding-dimension <embedding-dimension>]
```

Optional arguments:
- `--provider-id`: Provider ID for the vector db
- `--provider-vector-db-id`: Provider's vector db ID
- `--embedding-model`: Embedding model to use. Default: "all-MiniLM-L6-v2"
- `--embedding-dimension`: Dimension of embeddings. Default: 384

### `llama-stack-client vector_dbs unregister`
```bash
llama-stack-client vector_dbs unregister <vector-db-id>
```

## Shield Management
### `llama-stack-client shields list`
```bash
llama-stack-client shields list
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
llama-stack-client shields register --shield-id <shield-id> [--provider-id <provider-id>] [--provider-shield-id <provider-shield-id>] [--params <params>]
```

Required arguments:
- `--shield-id`: ID of the shield

Optional arguments:
- `--provider-id`: Provider ID for the shield
- `--provider-shield-id`: Provider's shield ID
- `--params`: JSON configuration parameters for the shield

## Eval Task Management

### `llama-stack-client benchmarks list`
```bash
llama-stack-client benchmarks list
```

### `llama-stack-client benchmarks register`
```bash
llama-stack-client benchmarks register --eval-task-id <eval-task-id> --dataset-id <dataset-id> --scoring-functions <function1> [<function2> ...] [--provider-id <provider-id>] [--provider-eval-task-id <provider-eval-task-id>] [--metadata <metadata>]
```

Required arguments:
- `--eval-task-id`: ID of the eval task
- `--dataset-id`: ID of the dataset to evaluate
- `--scoring-functions`: One or more scoring functions to use for evaluation

Optional arguments:
- `--provider-id`: Provider ID for the eval task
- `--provider-eval-task-id`: Provider's eval task ID
- `--metadata`: Metadata for the eval task in JSON format

## Eval execution
### `llama-stack-client eval run-benchmark`
```bash
llama-stack-client eval run-benchmark <eval-task-id1> [<eval-task-id2> ...] --eval-task-config <config-file> --output-dir <output-dir> [--num-examples <num>] [--visualize]
```

Required arguments:
- `--eval-task-config`: Path to the eval task config file in JSON format
- `--output-dir`: Path to the directory where evaluation results will be saved

Optional arguments:
- `--num-examples`: Number of examples to evaluate (useful for debugging)
- `--visualize`: If set, visualizes evaluation results after completion

Example benchmark_config.json:
```json
{
    "type": "benchmark",
    "eval_candidate": {
        "type": "model",
        "model": "Llama3.1-405B-Instruct",
        "sampling_params": {
            "strategy": "greedy",
        }
    }
}
```

### `llama-stack-client eval run-scoring`
```bash
llama-stack-client eval run-scoring <eval-task-id> --eval-task-config <config-file> --output-dir <output-dir> [--num-examples <num>] [--visualize]
```

Required arguments:
- `--eval-task-config`: Path to the eval task config file in JSON format
- `--output-dir`: Path to the directory where scoring results will be saved

Optional arguments:
- `--num-examples`: Number of examples to evaluate (useful for debugging)
- `--visualize`: If set, visualizes scoring results after completion

## Tool Group Management

### `llama-stack-client toolgroups list`
```bash
llama-stack-client toolgroups list
```
```
+---------------------------+------------------+------+---------------+
| identifier                | provider_id      | args | mcp_endpoint  |
+===========================+==================+======+===============+
| builtin::rag              | rag-runtime      | None | None          |
+---------------------------+------------------+------+---------------+
| builtin::websearch        | tavily-search    | None | None          |
+---------------------------+------------------+------+---------------+
```

### `llama-stack-client toolgroups get`
```bash
llama-stack-client toolgroups get <toolgroup_id>
```

Shows detailed information about a specific toolgroup. If the toolgroup is not found, displays an error message.

### `llama-stack-client toolgroups register`
```bash
llama-stack-client toolgroups register <toolgroup_id> [--provider-id <provider-id>] [--provider-toolgroup-id <provider-toolgroup-id>] [--mcp-config <mcp-config>] [--args <args>]
```

Optional arguments:
- `--provider-id`: Provider ID for the toolgroup
- `--provider-toolgroup-id`: Provider's toolgroup ID
- `--mcp-config`: JSON configuration for the MCP endpoint
- `--args`: JSON arguments for the toolgroup

### `llama-stack-client toolgroups unregister`
```bash
llama-stack-client toolgroups unregister <toolgroup_id>
```
