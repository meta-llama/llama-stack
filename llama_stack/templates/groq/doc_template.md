---
orphan: true
---
# Groq Distribution

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


### Prerequisite: API Keys

Make sure you have access to a Groq API Key. You can get one by visiting [Groq](https://api.groq.com/).


## Running Llama Stack with Groq

You can do this via Conda (build code) or Docker which has a pre-built image.

### Via Docker

This method allows you to get started quickly without having to build the distribution code.

```bash
LLAMA_STACK_PORT=8321
docker run \
  -it \
  --pull always \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  llamastack/distribution-{{ name }} \
  --port $LLAMA_STACK_PORT \
  --env GROQ_API_KEY=$GROQ_API_KEY
```

### Via Conda

```bash
llama stack build --template groq --image-type conda
llama stack run ./run.yaml \
  --port $LLAMA_STACK_PORT \
  --env GROQ_API_KEY=$GROQ_API_KEY
```
