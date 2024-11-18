# Ollama Distribution

The `llamastack/distribution-ollama` distribution consists of the following provider configurations.

| API | Provider(s) |
|-----|-------------|
| agents | `inline::meta-reference` |
| inference | `remote::ollama` |
| memory | `inline::faiss`, `remote::chromadb`, `remote::pgvector` |
| safety | `inline::llama-guard` |
| telemetry | `inline::meta-reference` |


You should use this distribution if you have a regular desktop machine without very powerful GPUs. Of course, if you have powerful GPUs, you can still continue using this distribution since Ollama supports GPU acceleration.### Models

The following models are configured by default:
- `${env.INFERENCE_MODEL}`
- `${env.SAFETY_MODEL}`

## Using Docker Compose

You can use `docker compose` to start a Ollama server and connect with Llama Stack server in a single command.

```bash
$ cd distributions/ollama; docker compose up
```

You will see outputs similar to following ---
```bash
[ollama]               | [GIN] 2024/10/18 - 21:19:41 | 200 |     226.841µs |             ::1 | GET      "/api/ps"
[ollama]               | [GIN] 2024/10/18 - 21:19:42 | 200 |      60.908µs |             ::1 | GET      "/api/ps"
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://[::]:5000 (Press CTRL+C to quit)
[llamastack] | Resolved 12 providers
[llamastack] |  inner-inference => ollama0
[llamastack] |  models => __routing_table__
[llamastack] |  inference => __autorouted__
```

To kill the server
```bash
docker compose down
```

## Starting Ollama and Llama Stack separately

If you wish to separately spin up a Ollama server, and connect with Llama Stack, you should use the following commands.

#### Start Ollama server
- Please check the [Ollama Documentation](https://github.com/ollama/ollama) for more details.

**Via Docker**
```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

**Via CLI**
```bash
ollama run <model_id>
```

#### Start Llama Stack server pointing to Ollama server

**Via Conda**

```bash
llama stack build --template ollama --image-type conda
llama stack run run.yaml
```

**Via Docker**
```
docker run --network host -it -p 5000:5000 -v ~/.llama:/root/.llama -v ./gpu/run.yaml:/root/llamastack-run-ollama.yaml --gpus=all llamastack/distribution-ollama --yaml_config /root/llamastack-run-ollama.yaml
```

Make sure in your `run.yaml` file, your inference provider is pointing to the correct Ollama endpoint. E.g.
```yaml
inference:
  - provider_id: ollama0
    provider_type: remote::ollama
    config:
      url: http://127.0.0.1:14343
```

### (Optional) Update Model Serving Configuration

#### Downloading model via Ollama

You can use ollama for managing model downloads.

```bash
ollama pull llama3.1:8b-instruct-fp16
ollama pull llama3.1:70b-instruct-fp16
```

> [!NOTE]
> Please check the [OLLAMA_SUPPORTED_MODELS](https://github.com/meta-llama/llama-stack/blob/main/llama_stack/providers.remote/inference/ollama/ollama.py) for the supported Ollama models.


To serve a new model with `ollama`
```bash
ollama run <model_name>
```

To make sure that the model is being served correctly, run `ollama ps` to get a list of models being served by ollama.
```
$ ollama ps

NAME                         ID              SIZE     PROCESSOR    UNTIL
llama3.1:8b-instruct-fp16    4aacac419454    17 GB    100% GPU     4 minutes from now
```

To verify that the model served by ollama is correctly connected to Llama Stack server
```bash
$ llama-stack-client models list
+----------------------+----------------------+---------------+-----------------------------------------------+
| identifier           | llama_model          | provider_id   | metadata                                      |
+======================+======================+===============+===============================================+
| Llama3.1-8B-Instruct | Llama3.1-8B-Instruct | ollama0       | {'ollama_model': 'llama3.1:8b-instruct-fp16'} |
+----------------------+----------------------+---------------+-----------------------------------------------+
```
