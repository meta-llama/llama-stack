---
orphan: true
---
# Together Distribution

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

Make sure you have access to a Together API Key. You can get one by visiting [together.xyz](https://together.xyz/).


## Running Llama Stack with Together

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
  --env TOGETHER_API_KEY=$TOGETHER_API_KEY
```

### Via Conda

```bash
llama stack build --template {{ name }} --image-type conda
llama stack run ./run.yaml \
  --port $LLAMA_STACK_PORT \
  --env TOGETHER_API_KEY=$TOGETHER_API_KEY
```
