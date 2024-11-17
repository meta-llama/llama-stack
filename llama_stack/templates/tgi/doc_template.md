# TGI Distribution

The `llamastack/distribution-{{ name }}` distribution consists of the following provider configurations.

{{ providers_table }}

You can use this distribution if you have GPUs and want to run an independent TGI server container for running inference.

{%- if docker_compose_env_vars %}
### Environment Variables

The following environment variables can be configured:

{% for var, (default_value, description) in docker_compose_env_vars.items() %}
- `{{ var }}`: {{ description }} (default: `{{ default_value }}`)
{% endfor %}
{% endif %}

{%- if default_models %}
### Models

The following models are configured by default:
{% for model in default_models %}
- `{{ model.model_id }}`
{% endfor %}
{% endif %}


## Using Docker Compose

You can use `docker compose` to start a TGI container and Llama Stack server container together.

```bash
$ cd distributions/{{ name }}; docker compose up
```

The script will first start up TGI server, then start up Llama Stack distribution server hooking up to the remote TGI provider for inference. You should be able to see the following outputs --
```bash
[text-generation-inference] | 2024-10-15T18:56:33.810397Z  INFO text_generation_router::server: router/src/server.rs:1813: Using config Some(Llama)
[text-generation-inference] | 2024-10-15T18:56:33.810448Z  WARN text_generation_router::server: router/src/server.rs:1960: Invalid hostname, defaulting to 0.0.0.0
[text-generation-inference] | 2024-10-15T18:56:33.864143Z  INFO text_generation_router::server: router/src/server.rs:2353: Connected
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://[::]:5001 (Press CTRL+C to quit)
```

To kill the server
```bash
docker compose down
```


### Conda: TGI server + llama stack run

If you wish to separately spin up a TGI server, and connect with Llama Stack, you may use the following commands.

#### Start TGI server locally
- Please check the [TGI Getting Started Guide](https://github.com/huggingface/text-generation-inference?tab=readme-ov-file#get-started) to get a TGI endpoint.

```bash
docker run --rm -it -v $HOME/.cache/huggingface:/data \
  -p 5009:5009 --gpus all \
  ghcr.io/huggingface/text-generation-inference:latest \
  --dtype bfloat16 --usage-stats on --sharded false \
  --model-id meta-llama/Llama-3.2-3B-Instruct --port 5009
```

#### Start Llama Stack server pointing to TGI server

**Via Conda**

```bash
llama stack build --template {{ name }} --image-type conda
# -- start a TGI server endpoint
llama stack run ./gpu/run.yaml
```

**Via Docker**
```bash
docker run --network host -it -p 5001:5001 \
  -v ./run.yaml:/root/my-run.yaml --gpus=all \
  llamastack/distribution-{{ name }} \
  --yaml_config /root/my-run.yaml
```

We have provided a template `run.yaml` file in the `distributions/{{ name }}` directory. Make sure in your `run.yaml` file, you inference provider is pointing to the correct TGI server endpoint. E.g.
```yaml
inference:
  - provider_id: tgi0
    provider_type: remote::tgi
    config:
      url: http://127.0.0.1:5009
```


### (Optional) Update Model Serving Configuration
To serve a new model with `tgi`, change the docker command flag `--model-id <model-to-serve>`.

This can be done by edit the `command` args in `compose.yaml`. E.g. Replace "Llama-3.2-1B-Instruct" with the model you want to serve.

```yaml
command: >
  --dtype bfloat16 --usage-stats on --sharded false
  --model-id meta-llama/Llama-3.2-1B-Instruct
  --port 5009 --cuda-memory-fraction 0.7
```

or by changing the docker run command's `--model-id` flag
```bash
docker run --rm -it -v $HOME/.cache/huggingface:/data \
  -p 5009:5009 --gpus all \
  ghcr.io/huggingface/text-generation-inference:latest \
  --dtype bfloat16 --usage-stats off --sharded false \
  --model-id meta-llama/Llama-3.2-3B-Instruct --port 5009
```

In `run.yaml`, make sure you point the correct server endpoint to the TGI server endpoint serving your model.
```yaml
inference:
  - provider_id: tgi0
    provider_type: remote::tgi
    config:
      url: http://127.0.0.1:5009
```
