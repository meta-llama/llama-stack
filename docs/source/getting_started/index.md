# Getting Started

```{toctree}
:hidden:
:maxdepth: 2

distributions/index
```

```{toctree}
:hidden:
developer_cookbook
```

At the end of the guide, you will have learnt how to:
- get a Llama Stack server up and running
- get a agent (with tool-calling, vector stores) which works with the above server

To see more example apps built using Llama Stack, see [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main).

## Step 1. Starting Up Llama Stack Server

### Decide Your Build Type
There are two ways to start a Llama Stack:

- **Docker**: we provide a number of pre-built Docker containers allowing you to get started instantly. If you are focused on application development, we recommend this option.
- **Conda**: the `llama` CLI provides a simple set of commands to build, configure and run a Llama Stack server containing the exact combination of providers you wish. We have provided various templates to make getting started easier.

Both of these provide options to run model inference using our reference implementations, Ollama, TGI, vLLM or even remote providers like Fireworks, Together, Bedrock, etc.

### Decide Your Inference Provider

Running inference of the underlying Llama model is one of the most critical requirements. Depending on what hardware you have available, you have various options. Note that each option have different necessary prerequisites.

- **Do you have access to a machine with powerful GPUs?**
If so, we suggest:
  - [`distribution-meta-reference-gpu`](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/meta-reference-gpu.html)
  - [`distribution-tgi`](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/tgi.html)

- **Are you running on a "regular" desktop machine?**
If so, we suggest:
  - [`distribution-ollama`](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/ollama.html)

- **Do you have access to a remote inference provider like Fireworks, Togther, etc.?** If so, we suggest:
  - [`distribution-together`](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/together.html)
  - [`distribution-fireworks`](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/fireworks.html)


### Quick Start Commands

The following quick starts commands. Please visit each distribution page on detailed setup.

##### 1.0 Prerequisite
::::{tab-set}

:::{tab-item} meta-reference-gpu
##### Downloading Models
Please make sure you have llama model checkpoints downloaded in `~/.llama` before proceeding. See [installation guide](https://llama-stack.readthedocs.io/en/latest/cli_reference/download_models.html) here to download the models.

```
$ ls ~/.llama/checkpoints
Llama3.1-8B           Llama3.2-11B-Vision-Instruct  Llama3.2-1B-Instruct  Llama3.2-90B-Vision-Instruct  Llama-Guard-3-8B
Llama3.1-8B-Instruct  Llama3.2-1B                   Llama3.2-3B-Instruct  Llama-Guard-3-1B              Prompt-Guard-86M
```
:::

:::{tab-item} tgi
This assumes you have access to GPU to start a TGI server with access to your GPU.
:::

::::

##### 1.1. Start the distribution

**Via Docker**
::::{tab-set}

:::{tab-item} meta-reference-gpu
```
$ cd distributions/meta-reference-gpu && docker compose up
```

> [!NOTE]
> This assumes you have access to GPU to start a local server with access to your GPU.


> [!NOTE]
> `~/.llama` should be the path containing downloaded weights of Llama models.


This will download and start running a pre-built docker container. Alternatively, you may use the following commands:

```
docker run -it -p 5000:5000 -v ~/.llama:/root/.llama -v ./run.yaml:/root/my-run.yaml --gpus=all distribution-meta-reference-gpu --yaml_config /root/my-run.yaml
```
:::

:::{tab-item} tgi
```
$ cd distributions/tgi/gpu && docker compose up
```

The script will first start up TGI server, then start up Llama Stack distribution server hooking up to the remote TGI provider for inference. You should be able to see the following outputs --
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

::::

**Via Conda**

::::{tab-set}

:::{tab-item} meta-reference-gpu
1. Install the `llama` CLI. See [CLI Reference](https://llama-stack.readthedocs.io/en/latest/cli_reference/index.html)

2. Build the `meta-reference-gpu` distribution

```
$ llama stack build --template meta-reference-gpu --image-type conda
```

3. Start running distribution
```
$ cd distributions/meta-reference-gpu
$ llama stack run ./run.yaml
```
:::

:::{tab-item} tgi
```bash
llama stack build --template tgi --image-type conda
# -- start a TGI server endpoint
llama stack run ./gpu/run.yaml
```
:::

::::


##### 1.2 (Optional) Serving Model
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

:::{tab-item} ollama
You can use ollama for managing model downloads.

```
ollama pull llama3.1:8b-instruct-fp16
ollama pull llama3.1:70b-instruct-fp16
```

> [!NOTE]
> Please check the [OLLAMA_SUPPORTED_MODELS](https://github.com/meta-llama/llama-stack/blob/main/llama_stack/providers/adapters/inference/ollama/ollama.py) for the supported Ollama models.


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

::::


## Step 2. Build Your Llama Stack App

### chat_completion sanity test
Once the server is setup, we can test it with a client to see the example outputs by . This will run the chat completion client and query the distribution’s `/inference/chat_completion` API. Send a POST request to the server:

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
