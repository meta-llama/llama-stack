# Nutanix Distribution

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
- `{{ model.model_id }} ({{ model.provider_model_id }})`
{% endfor %}
{% endif %}


### Prerequisite: API Keys
Make sure you have a Nutanix AI Endpoint deployed and a API key.  


## Running Llama Stack with Nutanix

You can do this via Conda (build code) or Docker.

### Via Docker

```bash
llama stack build --template nutanix --image-type docker

LLAMA_STACK_PORT=1740
llama stack run nutanix \
    --port $LLAMA_STACK_PORT \
    --env NUTANIX_API_KEY=$NUTANIX_API_KEY
```

### Via Conda

```bash
llama stack build --template nutanix --image-type conda

LLAMA_STACK_PORT=1740
llama stack run ./run.yaml \
    --port $LLAMA_STACK_PORT \
    --env NUTANIX_API_KEY=$NUTANIX_API_KEY
```
