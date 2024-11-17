# Remote vLLM Distribution

The `llamastack/distribution-{{ name }}` distribution consists of the following provider configurations:

{{ providers_table }}

You can use this distribution if you have GPUs and want to run an independent vLLM server container for running inference.

{%- if docker_compose_env_vars %}
### Environment Variables

The following environment variables can be configured:

{% for var, (default_value, description) in docker_compose_env_vars.items() %}
- `{{ var }}`: {{ description }} (default: `{{ default_value }}`)
{% endfor %}
{% endif %}

{% if default_models %}
### Models

The following models are configured by default:
{% for model in default_models %}
- `{{ model.model_id }}`
{% endfor %}
{% endif %}

## Using Docker Compose

You can use `docker compose` to start a vLLM container and Llama Stack server container together.
```bash
$ cd distributions/{{ name }}; docker compose up
```

You will see outputs similar to following ---
```
<TO BE FILLED>
```

To kill the server
```bash
docker compose down
```

## Starting vLLM and Llama Stack separately

You can also decide to start a vLLM server and connect with Llama Stack manually. There are two ways to start a vLLM server and connect with Llama Stack.

#### Start vLLM server.

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.2-3B-Instruct
```

Please check the [vLLM Documentation](https://docs.vllm.ai/en/v0.5.5/serving/deploying_with_docker.html) for more details.


#### Start Llama Stack server pointing to your vLLM server


We have provided a template `run.yaml` file in the `distributions/remote-vllm` directory. Please make sure to modify the `inference.provider_id` to point to your vLLM server endpoint. As an example, if your vLLM server is running on `http://127.0.0.1:8000`, your `run.yaml` file should look like the following:
```yaml
inference:
  - provider_id: vllm0
    provider_type: remote::vllm
    config:
      url: http://127.0.0.1:8000
```

**Via Conda**

If you are using Conda, you can build and run the Llama Stack server with the following commands:
```bash
cd distributions/remote-vllm
llama stack build --template remote-vllm --image-type conda
llama stack run run.yaml
```

**Via Docker**

You can use the Llama Stack Docker image to start the server with the following command:
```bash
docker run --network host -it -p 5000:5000 \
  -v ~/.llama:/root/.llama \
  -v ./gpu/run.yaml:/root/llamastack-run-remote-vllm.yaml \
  --gpus=all \
  llamastack/distribution-remote-vllm \
  --yaml_config /root/llamastack-run-remote-vllm.yaml
```
