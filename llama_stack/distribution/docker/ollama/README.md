# Ollama GPU Distribution

The scripts in these folders help you spin up a Llama Stack distribution with Ollama Inference provider.

> [!NOTE]
> This assumes you have access to GPU to start a Ollama server with access to your GPU. Please see Ollama CPU Distribution if you wish connect to a hosted Ollama endpoint.

### Getting Started

```
$ cd llama_stack/distribution/docker/ollama
$ ls
compose.yaml  ollama-run.yaml
$ docker compose up
```

You will see outputs similar to following ---
```
[ollama]               | [GIN] 2024/10/18 - 21:19:41 | 200 |     226.841µs |             ::1 | GET      "/api/ps"
[ollama]               | [GIN] 2024/10/18 - 21:19:42 | 200 |      60.908µs |             ::1 | GET      "/api/ps"
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://[::]:5000 (Press CTRL+C to quit)
[llamastack-local-cpu] | Resolved 12 providers
[llamastack-local-cpu] |  inner-inference => ollama0
[llamastack-local-cpu] |  models => __routing_table__
[llamastack-local-cpu] |  inference => __autorouted__
```

To kill the server
```
docker compose down
```

### (Alternative) Docker Run

If you wish to separately spin up a Ollama server, and connect with Llama Stack, you may use the following commands.

##### Start Ollama server.
- Please check the [Ollama Documentations](https://github.com/ollama/ollama) for more details.

**Via Docker**
```
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

**Via CLI**
```
ollama run <model_id>
```


##### Start Llama Stack server pointing to Ollama server

```
docker run --network host -it -p 5000:5000 -v ~/.llama:/root/.llama -v ./ollama-run.yaml:/root/llamastack-run-ollama.yaml --gpus=all llamastack-local-cpu --yaml_config /root/llamastack-run-ollama.yaml
```

Make sure in you `ollama-run.yaml` file, you inference provider is pointing to the correct Ollama endpoint. E.g.
```
inference:
  - provider_id: ollama0
    provider_type: remote::ollama
    config:
      url: http://127.0.0.1:14343
```
