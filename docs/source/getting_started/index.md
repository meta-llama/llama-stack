# Getting Started

```{toctree}
:maxdepth: 2
:hidden:

distributions/self_hosted_distro/index
distributions/remote_hosted_distro/index
distributions/ondevice_distro/index
```

At the end of the guide, you will have learned how to:
- get a Llama Stack server up and running
- set up an agent (with tool-calling and vector stores) that works with the above server

To see more example apps built using Llama Stack, see [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main).

## Step 1. Starting Up Llama Stack Server

### Decide Your Build Type
There are two ways to start a Llama Stack:

- **Docker**: we provide a number of pre-built Docker containers allowing you to get started instantly. If you are focused on application development, we recommend this option.
- **Conda**: the `llama` CLI provides a simple set of commands to build, configure and run a Llama Stack server containing the exact combination of providers you wish. We have provided various templates to make getting started easier.

Both of these provide options to run model inference using our reference implementations, Ollama, TGI, vLLM or even remote providers like Fireworks, Together, Bedrock, etc.

### Decide Your Inference Provider

Running inference on the underlying Llama model is one of the most critical requirements. Depending on what hardware you have available, you have various options. Note that each option have different necessary prerequisites.

- **Do you have access to a machine with powerful GPUs?**
If so, we suggest:
  - [distribution-meta-reference-gpu](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/self_hosted_distro/meta-reference-gpu.html)
  - [distribution-tgi](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/tgi.html)

- **Are you running on a "regular" desktop machine?**
If so, we suggest:
  - [distribution-ollama](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/self_hosted_distro/ollama.html)

- **Do you have an API key for a remote inference provider like Fireworks, Together, etc.?** If so, we suggest:
  - [distribution-together](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/remote_hosted_distro/together.html)
  - [distribution-fireworks](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/remote_hosted_distro/fireworks.html)

- **Do you want to run Llama Stack inference on your iOS / Android device** If so, we suggest:
  - [iOS](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/ondevice_distro/ios_sdk.html)
  - [Android](https://github.com/meta-llama/llama-stack-client-kotlin) (coming soon)

Please see our pages in detail for the types of distributions we offer:

1. [Self-Hosted Distribution](./distributions/self_hosted_distro/index.md): If you want to run Llama Stack inference on your local machine.
2. [Remote-Hosted Distribution](./distributions/remote_hosted_distro/index.md): If you want to connect to a remote hosted inference provider.
3. [On-device Distribution](./distributions/ondevice_distro/index.md): If you want to run Llama Stack inference on your iOS / Android device.


### Quick Start Commands

Once you have decided on the inference provider and distribution to use, use the following quick start commands to get started.

##### 1.0 Prerequisite

```
$ git clone git@github.com:meta-llama/llama-stack.git
```

::::{tab-set}

:::{tab-item} meta-reference-gpu
##### System Requirements
Access to Single-Node GPU to start a local server.

##### Downloading Models
Please make sure you have Llama model checkpoints downloaded in `~/.llama` before proceeding. See [installation guide](https://llama-stack.readthedocs.io/en/latest/cli_reference/download_models.html) here to download the models.

```
$ ls ~/.llama/checkpoints
Llama3.1-8B           Llama3.2-11B-Vision-Instruct  Llama3.2-1B-Instruct  Llama3.2-90B-Vision-Instruct  Llama-Guard-3-8B
Llama3.1-8B-Instruct  Llama3.2-1B                   Llama3.2-3B-Instruct  Llama-Guard-3-1B              Prompt-Guard-86M
```

:::

:::{tab-item} tgi
##### System Requirements
Access to Single-Node GPU to start a TGI server.
:::

:::{tab-item} ollama
##### System Requirements
Access to Single-Node CPU/GPU able to run ollama.
:::

:::{tab-item} together
##### System Requirements
Access to Single-Node CPU with Together hosted endpoint via API_KEY from [together.ai](https://api.together.xyz/signin).
:::

:::{tab-item} fireworks
##### System Requirements
Access to Single-Node CPU with Fireworks hosted endpoint via API_KEY from [fireworks.ai](https://fireworks.ai/).
:::

::::

##### 1.1. Start the distribution

**(Option 1) Via Docker**
::::{tab-set}

:::{tab-item} meta-reference-gpu
```
$ cd llama-stack/distributions/meta-reference-gpu && docker compose up
```

This will download and start running a pre-built Docker container. Alternatively, you may use the following commands:

```
docker run -it -p 5000:5000 -v ~/.llama:/root/.llama -v ./run.yaml:/root/my-run.yaml --gpus=all distribution-meta-reference-gpu --yaml_config /root/my-run.yaml
```
:::

:::{tab-item} tgi
```
$ cd llama-stack/distributions/tgi/gpu && docker compose up
```

The script will first start up TGI server, then start up Llama Stack distribution server hooking up to the remote TGI provider for inference. You should see the following outputs --
```
[text-generation-inference] | 2024-10-15T18:56:33.810397Z  INFO text_generation_router::server: router/src/server.rs:1813: Using config Some(Llama)
[text-generation-inference] | 2024-10-15T18:56:33.810448Z  WARN text_generation_router::server: router/src/server.rs:1960: Invalid hostname, defaulting to 0.0.0.0
[text-generation-inference] | 2024-10-15T18:56:33.864143Z  INFO text_generation_router::server: router/src/server.rs:2353: Connected
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://[::]:5000 (Press CTRL+C to quit)
```

To kill the server
```
docker compose down
```
:::


:::{tab-item} ollama
```
$ cd llama-stack/distributions/ollama/cpu && docker compose up
```

You will see outputs similar to following ---
```
[ollama]               | [GIN] 2024/10/18 - 21:19:41 | 200 |     226.841µs |             ::1 | GET      "/api/ps"
[ollama]               | [GIN] 2024/10/18 - 21:19:42 | 200 |      60.908µs |             ::1 | GET      "/api/ps"
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://[::]:5000 (Press CTRL+C to quit)
[llamastack] | Resolved 12 providers
[llamastack] |  inner-inference => ollama0
[llamastack] |  models => __routing_table__
[llamastack] |  inference => __autorouted__
```

To kill the server
```
docker compose down
```
:::

:::{tab-item} fireworks
```
$ cd llama-stack/distributions/fireworks && docker compose up
```

Make sure your `run.yaml` file has the inference provider pointing to the correct Fireworks URL server endpoint. E.g.
```
inference:
  - provider_id: fireworks
    provider_type: remote::fireworks
    config:
      url: https://api.fireworks.ai/inference
      api_key: <optional api key>
```
:::

:::{tab-item} together
```
$ cd distributions/together && docker compose up
```

Make sure your `run.yaml` file has the inference provider pointing to the correct Together URL server endpoint. E.g.
```
inference:
  - provider_id: together
    provider_type: remote::together
    config:
      url: https://api.together.xyz/v1
      api_key: <optional api key>
```
:::


::::

**(Option 2) Via Conda**

::::{tab-set}

:::{tab-item} meta-reference-gpu
1. Install the `llama` CLI. See [CLI Reference](https://llama-stack.readthedocs.io/en/latest/cli_reference/index.html)

2. Build the `meta-reference-gpu` distribution

```
$ llama stack build --template meta-reference-gpu --image-type conda
```

3. Start running distribution
```
$ cd llama-stack/distributions/meta-reference-gpu
$ llama stack run ./run.yaml
```
:::

:::{tab-item} tgi
1. Install the `llama` CLI. See [CLI Reference](https://llama-stack.readthedocs.io/en/latest/cli_reference/index.html)

2. Build the `tgi` distribution

```bash
llama stack build --template tgi --image-type conda
```

3. Start a TGI server endpoint

4. Make sure in your `run.yaml` file, your `conda_env` is pointing to the conda environment and inference provider is pointing to the correct TGI server endpoint. E.g.
```
conda_env: llamastack-tgi
...
inference:
  - provider_id: tgi0
    provider_type: remote::tgi
    config:
      url: http://127.0.0.1:5009
```

5. Start Llama Stack server
```bash
llama stack run ./gpu/run.yaml
```
:::

:::{tab-item} ollama

If you wish to separately spin up a Ollama server, and connect with Llama Stack, you may use the following commands.

#### Start Ollama server.
- Please check the [Ollama Documentations](https://github.com/ollama/ollama) for more details.

**Via Docker**
```
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

**Via CLI**
```
ollama run <model_id>
```

#### Start Llama Stack server pointing to Ollama server

Make sure your `run.yaml` file has the inference provider pointing to the correct Ollama endpoint. E.g.
```
conda_env: llamastack-ollama
...
inference:
  - provider_id: ollama0
    provider_type: remote::ollama
    config:
      url: http://127.0.0.1:11434
```

```
llama stack build --template ollama --image-type conda
llama stack run ./gpu/run.yaml
```

:::

:::{tab-item} fireworks

```bash
llama stack build --template fireworks --image-type conda
# -- modify run.yaml to a valid Fireworks server endpoint
llama stack run ./run.yaml
```

Make sure your `run.yaml` file has the inference provider pointing to the correct Fireworks URL server endpoint. E.g.
```
conda_env: llamastack-fireworks
...
inference:
  - provider_id: fireworks
    provider_type: remote::fireworks
    config:
      url: https://api.fireworks.ai/inference
      api_key: <optional api key>
```
:::

:::{tab-item} together

```bash
llama stack build --template together --image-type conda
# -- modify run.yaml to a valid Together server endpoint
llama stack run ./run.yaml
```

Make sure your `run.yaml` file has the inference provider pointing to the correct Together URL server endpoint. E.g.
```
conda_env: llamastack-together
...
inference:
  - provider_id: together
    provider_type: remote::together
    config:
      url: https://api.together.xyz/v1
      api_key: <optional api key>
```
:::

::::

##### 1.2 (Optional) Update Model Serving Configuration
::::{tab-set}

:::{tab-item} meta-reference-gpu
You may change the `config.model` in `run.yaml` to update the model currently being served by the distribution. Make sure you have the model checkpoint downloaded in your `~/.llama`.
```
inference:
  - provider_id: meta0
    provider_type: meta-reference
    config:
      model: Llama3.2-11B-Vision-Instruct
      quantization: null
      torch_seed: null
      max_seq_len: 4096
      max_batch_size: 1
```

Run `llama model list` to see the available models to download, and `llama model download` to download the checkpoints.
:::

:::{tab-item} tgi
To serve a new model with `tgi`, change the docker command flag `--model-id <model-to-serve>`.

This can be done by edit the `command` args in `compose.yaml`. E.g. Replace "Llama-3.2-1B-Instruct" with the model you want to serve.

```
command: ["--dtype", "bfloat16", "--usage-stats", "on", "--sharded", "false", "--model-id", "meta-llama/Llama-3.2-1B-Instruct", "--port", "5009", "--cuda-memory-fraction", "0.3"]
```

or by changing the docker run command's `--model-id` flag
```
docker run --rm -it -v $HOME/.cache/huggingface:/data -p 5009:5009 --gpus all ghcr.io/huggingface/text-generation-inference:latest --dtype bfloat16 --usage-stats on --sharded false --model-id meta-llama/Llama-3.2-1B-Instruct --port 5009
```

Make sure your `run.yaml` file has the inference provider pointing to the TGI server endpoint serving your model.
```
inference:
  - provider_id: tgi0
    provider_type: remote::tgi
    config:
      url: http://127.0.0.1:5009
```
```

Run `llama model list` to see the available models to download, and `llama model download` to download the checkpoints.
:::

:::{tab-item} ollama
You can use ollama for managing model downloads.

```
ollama pull llama3.1:8b-instruct-fp16
ollama pull llama3.1:70b-instruct-fp16
```

> Please check the [OLLAMA_SUPPORTED_MODELS](https://github.com/meta-llama/llama-stack/blob/main/llama_stack/providers.remote/inference/ollama/ollama.py) for the supported Ollama models.


To serve a new model with `ollama`
```
ollama run <model_name>
```

To make sure that the model is being served correctly, run `ollama ps` to get a list of models being served by ollama.
```
$ ollama ps

NAME                         ID              SIZE     PROCESSOR    UNTIL
llama3.1:8b-instruct-fp16    4aacac419454    17 GB    100% GPU     4 minutes from now
```

To verify that the model served by ollama is correctly connected to Llama Stack server
```
$ llama-stack-client models list
+----------------------+----------------------+---------------+-----------------------------------------------+
| identifier           | llama_model          | provider_id   | metadata                                      |
+======================+======================+===============+===============================================+
| Llama3.1-8B-Instruct | Llama3.1-8B-Instruct | ollama0       | {'ollama_model': 'llama3.1:8b-instruct-fp16'} |
+----------------------+----------------------+---------------+-----------------------------------------------+
```
:::

:::{tab-item} together
Use `llama-stack-client models list` to check the available models served by together.

```
$ llama-stack-client models list
+------------------------------+------------------------------+---------------+------------+
| identifier                   | llama_model                  | provider_id   | metadata   |
+==============================+==============================+===============+============+
| Llama3.1-8B-Instruct         | Llama3.1-8B-Instruct         | together0     | {}         |
+------------------------------+------------------------------+---------------+------------+
| Llama3.1-70B-Instruct        | Llama3.1-70B-Instruct        | together0     | {}         |
+------------------------------+------------------------------+---------------+------------+
| Llama3.1-405B-Instruct       | Llama3.1-405B-Instruct       | together0     | {}         |
+------------------------------+------------------------------+---------------+------------+
| Llama3.2-3B-Instruct         | Llama3.2-3B-Instruct         | together0     | {}         |
+------------------------------+------------------------------+---------------+------------+
| Llama3.2-11B-Vision-Instruct | Llama3.2-11B-Vision-Instruct | together0     | {}         |
+------------------------------+------------------------------+---------------+------------+
| Llama3.2-90B-Vision-Instruct | Llama3.2-90B-Vision-Instruct | together0     | {}         |
+------------------------------+------------------------------+---------------+------------+
```
:::

:::{tab-item} fireworks
Use `llama-stack-client models list` to check the available models served by Fireworks.
```
$ llama-stack-client models list
+------------------------------+------------------------------+---------------+------------+
| identifier                   | llama_model                  | provider_id   | metadata   |
+==============================+==============================+===============+============+
| Llama3.1-8B-Instruct         | Llama3.1-8B-Instruct         | fireworks0    | {}         |
+------------------------------+------------------------------+---------------+------------+
| Llama3.1-70B-Instruct        | Llama3.1-70B-Instruct        | fireworks0    | {}         |
+------------------------------+------------------------------+---------------+------------+
| Llama3.1-405B-Instruct       | Llama3.1-405B-Instruct       | fireworks0    | {}         |
+------------------------------+------------------------------+---------------+------------+
| Llama3.2-1B-Instruct         | Llama3.2-1B-Instruct         | fireworks0    | {}         |
+------------------------------+------------------------------+---------------+------------+
| Llama3.2-3B-Instruct         | Llama3.2-3B-Instruct         | fireworks0    | {}         |
+------------------------------+------------------------------+---------------+------------+
| Llama3.2-11B-Vision-Instruct | Llama3.2-11B-Vision-Instruct | fireworks0    | {}         |
+------------------------------+------------------------------+---------------+------------+
| Llama3.2-90B-Vision-Instruct | Llama3.2-90B-Vision-Instruct | fireworks0    | {}         |
+------------------------------+------------------------------+---------------+------------+
```
:::

::::


##### Troubleshooting
- If you encounter any issues, search through our [GitHub Issues](https://github.com/meta-llama/llama-stack/issues), or file an new issue.
- Use `--port <PORT>` flag to use a different port number. For docker run, update the `-p <PORT>:<PORT>` flag.


## Step 2. Run Llama Stack App

### Chat Completion Test
Once the server is set up, we can test it with a client to verify it's working correctly. The following command will send a chat completion request to the server's `/inference/chat_completion` API:

```bash
$ curl http://localhost:5000/inference/chat_completion \
-H "Content-Type: application/json" \
-d '{
    "model": "Llama3.1-8B-Instruct",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write me a 2 sentence poem about the moon"}
    ],
    "sampling_params": {"temperature": 0.7, "seed": 42, "max_tokens": 512}
}'

Output:
{'completion_message': {'role': 'assistant',
  'content': 'The moon glows softly in the midnight sky, \nA beacon of wonder, as it catches the eye.',
  'stop_reason': 'out_of_tokens',
  'tool_calls': []},
 'logprobs': null}

```

### Run Agent App

To run an agent app, check out examples demo scripts with client SDKs to talk with the Llama Stack server in our [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repo. To run a simple agent app:

```bash
$ git clone git@github.com:meta-llama/llama-stack-apps.git
$ cd llama-stack-apps
$ pip install -r requirements.txt

$ python -m examples.agents.client <host> <port>
```

You will see outputs of the form --
```
User> I am planning a trip to Switzerland, what are the top 3 places to visit?
inference> Switzerland is a beautiful country with a rich history, stunning landscapes, and vibrant culture. Here are three must-visit places to add to your itinerary:
...

User> What is so special about #1?
inference> Jungfraujoch, also known as the "Top of Europe," is a unique and special place for several reasons:
...

User> What other countries should I consider to club?
inference> Considering your interest in Switzerland, here are some neighboring countries that you may want to consider visiting:
```
