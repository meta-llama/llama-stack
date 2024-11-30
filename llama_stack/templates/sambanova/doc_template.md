# Sambanova Distribution

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

Make sure you have access to a Sambanova API Key. You can get one by visiting [Sambanova](https://cloud.sambanova.ai/apis).

## Running Llama Stack with Sambanova

You can do this via Conda (build code).

### Available INFERENCE_MODEL

- meta-llama/Llama-3.1-8B-Instruct
- meta-llama/Llama-3.1-70B-Instruct
- meta-llama/Llama-3.1-405B-Instruct
- meta-llama/Llama-3.2-1B-Instruct
- meta-llama/Llama-3.2-3B-Instruct

### Via Conda

```bash
llama stack build --template sambanova --image-type conda

conda activate llamastack-sambanova

export SAMBANOVA_API_KEY={YOUR_API_KEY}
export INFERENCE_MODEL={CHOOSE_AND_FIND_AVAILABLE_MODEL_ABOVE}

llama stack run \
  --port $LLAMA_STACK_PORT \
  --env SAMBANOVA_API_KEY=$SAMBANOVA_API_KEY \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  sambanova
```
