---
orphan: true
---
# RamaLama Distribution

```{toctree}
:maxdepth: 2
:hidden:

self
```

The `llamastack/distribution-{{ name }}` distribution consists of the following provider configurations.

{{ providers_table }}

You should use this distribution if you have a regular desktop machine without very powerful GPUs. Of course, if you have powerful GPUs, you can still continue using this distribution since RamaLama supports GPU acceleration.

{% if run_config_env_vars %}
### Environment Variables

The following environment variables can be configured:

{% for var, (default_value, description) in run_config_env_vars.items() %}
- `{{ var }}`: {{ description }} (default: `{{ default_value }}`)
{% endfor %}
{% endif %}


## Setting up RamaLama server

Please check the [RamaLama Documentation](https://github.com/containers/ramalama) on how to install and run RamaLama. After installing RamaLama, you need to run `ramalama serve` to start the server.

In order to load models, you can run:

```bash
export INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"

# ramalama names this model differently, and we must use the ramalama name when loading the model
export OLLAMA_INFERENCE_MODEL="llama3.2:3b-instruct-fp16"
ramalama run $OLLAMA_INFERENCE_MODEL --keepalive 60m
```

If you are using Llama Stack Safety / Shield APIs, you will also need to pull and run the safety model.

```bash
export SAFETY_MODEL="meta-llama/Llama-Guard-3-1B"

# ramalama names this model differently, and we must use the ramalama name when loading the model
export OLLAMA_SAFETY_MODEL="llama-guard3:1b"
ramalama run $OLLAMA_SAFETY_MODEL --keepalive 60m
```

## Running Llama Stack

Now you are ready to run Llama Stack with RamaLama as the inference provider. You can do this via Conda (build code) or Docker which has a pre-built image.

### Via Podman

This method allows you to get started quickly without having to build the distribution code.

```bash
export LLAMA_STACK_PORT=5001
podman run \
  -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama:z \
  llamastack/distribution-{{ name }} \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env OLLAMA_URL=http://host.containers.internal:11434
```

If you are using Llama Stack Safety / Shield APIs, use:

```bash
# You need a local checkout of llama-stack to run this, get it using
# git clone https://github.com/meta-llama/llama-stack.git
cd /path/to/llama-stack

podman run \
  -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama:z \
  -v ./llama_stack/templates/ramalama/run-with-safety.yaml:/root/my-run.yaml:z \
  llamastack/distribution-{{ name }} \
  --yaml-config /root/my-run.yaml \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env SAFETY_MODEL=$SAFETY_MODEL \
  --env OLLAMA_URL=http://host.containers.internal:11434
```

### Via Docker

This method allows you to get started quickly without having to build the distribution code.

```bash
export LLAMA_STACK_PORT=5001
docker run \
  -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  llamastack/distribution-{{ name }} \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env OLLAMA_URL=http://host.docker.internal:11434
```

If you are using Llama Stack Safety / Shield APIs, use:

```bash
# You need a local checkout of llama-stack to run this, get it using
# git clone https://github.com/meta-llama/llama-stack.git
cd /path/to/llama-stack

docker run \
  -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  -v ./llama_stack/templates/ramalama/run-with-safety.yaml:/root/my-run.yaml \
  llamastack/distribution-{{ name }} \
  --yaml-config /root/my-run.yaml \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env SAFETY_MODEL=$SAFETY_MODEL \
  --env OLLAMA_URL=http://host.docker.internal:11434
```

### Via Conda

Make sure you have done `uv pip install llama-stack` and have the Llama Stack CLI available.

```bash
export LLAMA_STACK_PORT=5001

llama stack build --template {{ name }} --image-type conda
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
Please check the [model_aliases](https://github.com/meta-llama/llama-stack/blob/main/llama_stack/providers/remote/inference/ramalama/ramalama.py#L45) for the supported RamaLama models.
```

To serve a new model with `ramalama`
```bash
ramalama run <model_name>
```

To make sure that the model is being served correctly, run `ramalama ps` to get a list of models being served by ramalama.
```
$ ramalama ps

NAME                         ID              SIZE     PROCESSOR    UNTIL
llama3.1:8b-instruct-fp16    4aacac419454    17 GB    100% GPU     4 minutes from now
```

To verify that the model served by ramalama is correctly connected to Llama Stack server
```bash
$ llama-stack-client models list
+----------------------+----------------------+---------------+-----------------------------------------------+
| identifier           | llama_model          | provider_id   | metadata                                      |
+======================+======================+===============+===============================================+
| Llama3.1-8B-Instruct | Llama3.1-8B-Instruct | ramalama0       | {'ramalama_model': 'llama3.1:8b-instruct-fp16'} |
+----------------------+----------------------+---------------+-----------------------------------------------+
```
