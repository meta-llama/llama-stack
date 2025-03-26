---
orphan: true
---
# Podman AI Lab Distribution

```{toctree}
:maxdepth: 2
:hidden:

self
```

The `llamastack/distribution-{{ name }}` distribution consists of the following provider configurations.

{{ providers_table }}

You should use this distribution if you have a regular desktop machine without very powerful GPUs. Of course, if you have powerful GPUs, you can still continue using this distribution since Podman AI Lab supports GPU acceleration.

{% if run_config_env_vars %}
### Environment Variables

The following environment variables can be configured:

{% for var, (default_value, description) in run_config_env_vars.items() %}
- `{{ var }}`: {{ description }} (default: `{{ default_value }}`)
{% endfor %}
{% endif %}


## Setting up Podman AI Lab server

Please check the [Podman AI Lab Documentation](https://github.com/containers/podman-desktop-extension-ai-lab) on how to install and run Podman AI Lab.


If you are using Llama Stack Safety / Shield APIs, you will also need to pull and run the safety model.

```bash
export SAFETY_MODEL="meta-llama/Llama-Guard-3-1B"

export PODMAN_AI_LAB_SAFETY_MODEL="llama-guard3:1b"
```

## Running Llama Stack

Now you are ready to run Llama Stack with Podman AI Lab as the inference provider. You can do this via Conda (build code) or Docker which has a pre-built image.

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
  --env PODMAN_AI_LAB_URL=http://host.docker.internal:10434
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
  -v ./llama_stack/templates/podman-ai-lab/run-with-safety.yaml:/root/my-run.yaml \
  llamastack/distribution-{{ name }} \
  --yaml-config /root/my-run.yaml \
  --port $LLAMA_STACK_PORT \
  --env SAFETY_MODEL=$SAFETY_MODEL \
  --env PODMAN_AI_LAB_URL=http://host.docker.internal:11434
```

### Via Conda

Make sure you have done `uv pip install llama-stack` and have the Llama Stack CLI available.

```bash
export LLAMA_STACK_PORT=5001

llama stack build --template {{ name }} --image-type conda
llama stack run ./run.yaml \
  --port $LLAMA_STACK_PORT \
  --env PODMAN_AI_LAB_URL=http://localhost:10434
```

If you are using Llama Stack Safety / Shield APIs, use:

```bash
llama stack run ./run-with-safety.yaml \
  --port $LLAMA_STACK_PORT \
  --env SAFETY_MODEL=$SAFETY_MODEL \
  --env PODMAN_AI_LAB_URL=http://localhost:11434
```


### (Optional) Update Model Serving Configuration

To serve a new model with `Podman AI Lab`:
- launch Podman Desktop with Podman AI Lab extension installed
- download the model
- start an inference server for the model

To make sure that the model is being served correctly, run `curl localhost:10434/api/tags` to get a list of models being served by Podman AI Lab.
```
$ curl localhost:10434/api/tags
{"models":[{"model":"hf.ibm-research.granite-3.2-8b-instruct-GGUF","name":"ibm-research/granite-3.2-8b-instruct-GGUF","digest":"363f0bbc3200b9c9b0ab87efe237d77b1e05bb929d5d7e4b57c1447c911223e8","size":4942859552,"modified_at":"2025-03-17T14:48:32.417Z","details":{}}]}
```

To verify that the model served by Podman AI Lab is correctly connected to Llama Stack server
```bash
$ llama-stack-client models list

Available Models

┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ model_type   ┃ identifier                                     ┃ provider_resource_id                          ┃ metadata  ┃ provider_id    ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ llm          │ ibm-research/granite-3.2-8b-instruct-GGUF      │ ibm-research/granite-3.2-8b-instruct-GGUF     │           │ podman-ai-lab  │
└──────────────┴────────────────────────────────────────────────┴───────────────────────────────────────────────┴───────────┴────────────────┘

Total models: 1
```
