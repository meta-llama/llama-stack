# Ssambanova Distribution

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

Make sure you have access to a Ssambanova API Key. You can get one by visiting [Ssambanova](https://cloud.sambanova.ai/apis).

## Running Llama Stack with Ssambanova

You can do this via Conda (build code) or Docker which has a pre-built image.

### Available INFERENCE_MODEL

- Meta-Llama-3.1-8B-Instruct
- Meta-Llama-3.1-70B-Instruct
- Meta-Llama-3.1-405B-Instruct
- Meta-Llama-3.2-1B-Instruct
- Meta-Llama-3.2-3B-Instruct

### Via Docker

This method allows you to get started quickly without having to build the distribution code.

```bash
LLAMA_STACK_PORT=5001
docker run \
  -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  llamastack/distribution-{{ name }} \
  --port $LLAMA_STACK_PORT \
  --env SSAMBANOVA_API_KEY=$SSAMBANOVA_API_KEY \
  --env INFERENCE_MODEL=$INFERENCE_MODEL
```

### Via Conda

```bash
llama stack build --template ssambanova --image-type conda
llama stack run ./run.yaml \
  --port $LLAMA_STACK_PORT \
  --env SSAMBANOVA_API_KEY=$SSAMBANOVA_API_KEY \
  --env INFERENCE_MODEL=$INFERENCE_MODEL
```
