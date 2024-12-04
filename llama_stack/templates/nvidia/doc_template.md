# NVIDIA Distribution

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

Make sure you have access to a NVIDIA API Key. You can get one by visiting [https://build.nvidia.com/](https://build.nvidia.com/).


## Running Llama Stack with NVIDIA

You can do this via Conda (build code) or Docker which has a pre-built image.

### Via Docker

This method allows you to get started quickly without having to build the distribution code.

```bash
LLAMA_STACK_PORT=5001
docker run \
  -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ./run.yaml:/root/my-run.yaml \
  llamastack/distribution-{{ name }} \
  --yaml-config /root/my-run.yaml \
  --port $LLAMA_STACK_PORT \
  --env NVIDIA_API_KEY=$NVIDIA_API_KEY
```

### Via Conda

```bash
llama stack build --template nvidia --image-type conda
llama stack run ./run.yaml \
  --port 5001 \
  --env NVIDIA_API_KEY=$NVIDIA_API_KEY
```
