---
orphan: true
---
# Clarifai Distribution

```{toctree}
:maxdepth: 2
:hidden:

self
```

The `llamastack/distribution-{{ name }}` distribution consists of the following provider configurations.

{{ providers_table }}

{% if run_config_env_vars %}
### Environment Variables

The following environment variables can be configured:

{% for var, (default_value, description) in run_config_env_vars.items() %}
- `{{ var }}`: {{ description }} (default: `{{ default_value }}`)
{% endfor %}
{% endif %}

{% if default_models %}
### Models

The following models are available by default:

{% for model in default_models %}
- `{{ model.model_id }} {{ model.doc_string }}`
{% endfor %}
{% endif %}


### Prerequisite: PAT

Make sure you have access to a Clarifai PAT. You can get one by visiting [Clarifai](https://www.clarifai.com/).


## Running Llama Stack with Clarifai

You can do this via Conda (build code) or Docker which has a pre-built image.

### Via Docker

This method allows you to get started quickly without having to build the distribution code.

```bash
LLAMA_STACK_PORT=5001
docker run \
  -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  llamastack/distribution-{{ name }} \
  --port $LLAMA_STACK_PORT \
  --env CLARIFAI_PAT=$CLARIFAI_PAT
```

### Via Conda

```bash
llama stack build --template clarifai --image-type conda
llama stack run ./run.yaml \
  --port $LLAMA_STACK_PORT \
  --env CLARIFAI_PAT=$CLARIFAI_PAT
```
