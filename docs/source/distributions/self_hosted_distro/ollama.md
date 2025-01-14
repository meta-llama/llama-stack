---
orphan: true
---
# Ollama Distribution

```{toctree}
:maxdepth: 2
:hidden:

self
```

The `llamastack/distribution-ollama` distribution consists of the following provider configurations.

| API | Provider(s) |
|-----|-------------|
| agents | `inline::meta-reference` |
| datasetio | `remote::huggingface`, `inline::localfs` |
| eval | `inline::meta-reference` |
| inference | `remote::ollama` |
| memory | `inline::faiss`, `remote::chromadb`, `remote::pgvector` |
| safety | `inline::llama-guard` |
| scoring | `inline::basic`, `inline::llm-as-judge`, `inline::braintrust` |
| telemetry | `inline::meta-reference` |
| tool_runtime | `remote::brave-search`, `remote::tavily-search`, `inline::code-interpreter`, `inline::memory-runtime` |


You should use this distribution if you have a regular desktop machine without very powerful GPUs. Of course, if you have powerful GPUs, you can still continue using this distribution since Ollama supports GPU acceleration.### Environment Variables

The following environment variables can be configured:

- `LLAMA_STACK_PORT`: Port for the Llama Stack distribution server (default: `5001`)
- `OLLAMA_URL`: URL of the Ollama server (default: `http://127.0.0.1:11434`)
- `INFERENCE_MODEL`: Inference model loaded into the Ollama server (default: `meta-llama/Llama-3.2-3B-Instruct`)
- `SAFETY_MODEL`: Safety model loaded into the Ollama server (default: `meta-llama/Llama-Guard-3-1B`)


## Setting up Ollama server

Please check the [Ollama Documentation](https://github.com/ollama/ollama) on how to install and run Ollama. After installing Ollama, you need to run `ollama serve` to start the server.

In order to load models, you can run:

```bash
export INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"

# ollama names this model differently, and we must use the ollama name when loading the model
export OLLAMA_INFERENCE_MODEL="llama3.2:3b-instruct-fp16"
ollama run $OLLAMA_INFERENCE_MODEL --keepalive 60m
```

If you are using Llama Stack Safety / Shield APIs, you will also need to pull and run the safety model.

```bash
export SAFETY_MODEL="meta-llama/Llama-Guard-3-1B"

# ollama names this model differently, and we must use the ollama name when loading the model
export OLLAMA_SAFETY_MODEL="llama-guard3:1b"
ollama run $OLLAMA_SAFETY_MODEL --keepalive 60m
```

## Running Llama Stack

Now you are ready to run Llama Stack with Ollama as the inference provider. You can do this via Conda (build code) or Docker which has a pre-built image.

### Via Docker

This method allows you to get started quickly without having to build the distribution code.

```bash
export LLAMA_STACK_PORT=5001
docker run \
  -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  llamastack/distribution-ollama \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env OLLAMA_URL=http://host.docker.internal:11434
```

If you are using Llama Stack Safety / Shield APIs, use:

```bash
docker run \
  -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  -v ./run-with-safety.yaml:/root/my-run.yaml \
  llamastack/distribution-ollama \
  --yaml-config /root/my-run.yaml \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env SAFETY_MODEL=$SAFETY_MODEL \
  --env OLLAMA_URL=http://host.docker.internal:11434
```

### Via Conda

Make sure you have done `pip install llama-stack` and have the Llama Stack CLI available.

```bash
export LLAMA_STACK_PORT=5001

llama stack build --template ollama --image-type conda
llama stack run ./run.yaml \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env OLLAMA_URL=http://localhost:11434
```

If you are using Llama Stack Safety / Shield APIs, use:

```bash
llama stack run ./run-with-safety.yaml \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env SAFETY_MODEL=$SAFETY_MODEL \
  --env OLLAMA_URL=http://localhost:11434
```


### (Optional) Update Model Serving Configuration

```{note}
Please check the [model_aliases](https://github.com/meta-llama/llama-stack/blob/main/llama_stack/providers/remote/inference/ollama/ollama.py#L45) for the supported Ollama models.
```

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
