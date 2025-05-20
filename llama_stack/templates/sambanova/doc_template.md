---
orphan: true
---
# SambaNova Distribution

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

Make sure you have access to a SambaNova API Key. You can get one by visiting [SambaNova.ai](http://cloud.sambanova.ai?utm_source=llamastack&utm_medium=external&utm_campaign=cloud_signup).


## Running Llama Stack with SambaNova

You can do this via Conda (build code) or Docker which has a pre-built image.


### Via Docker

```bash
LLAMA_STACK_PORT=8321
llama stack build --template sambanova --image-type container
docker run \
  -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  distribution-{{ name }} \
  --port $LLAMA_STACK_PORT \
  --env SAMBANOVA_API_KEY=$SAMBANOVA_API_KEY
```


### Via Venv

```bash
llama stack build --template sambanova --image-type venv
llama stack run --image-type venv ~/.llama/distributions/sambanova/sambanova-run.yaml \
  --port $LLAMA_STACK_PORT \
  --env SAMBANOVA_API_KEY=$SAMBANOVA_API_KEY
```


### Via Conda

```bash
llama stack build --template sambanova --image-type conda
llama stack run --image-type conda ~/.llama/distributions/sambanova/sambanova-run.yaml \
  --port $LLAMA_STACK_PORT \
  --env SAMBANOVA_API_KEY=$SAMBANOVA_API_KEY
```
