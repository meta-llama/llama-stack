---
orphan: true
---
# CentML Distribution

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
- `{{ model.model_id }}`
{% endfor %}
{% endif %}

### Prerequisite: API Keys

Make sure you have a valid **CentML API Key**. Sign up or access your credentials at [CentML.com](https://centml.com/).

## Running Llama Stack with CentML

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
  --env CENTML_API_KEY=$CENTML_API_KEY
```

### Via Conda

```bash
llama stack build --template {{ name }} --image-type conda
llama stack run ./run.yaml \
  --port $LLAMA_STACK_PORT \
  --env CENTML_API_KEY=$CENTML_API_KEY
```