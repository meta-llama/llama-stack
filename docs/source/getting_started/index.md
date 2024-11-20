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


### Table of Contents

Once you have decided on the inference provider and distribution to use, use the following guides to get started.

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

:::{tab-item} vLLM
##### System Requirements
Access to Single-Node GPU to start a vLLM server.
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

::::{tab-set}
:::{tab-item} meta-reference-gpu
- [Start Meta Reference GPU Distribution](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/self_hosted_distro/meta-reference-gpu.html)
:::

:::{tab-item} vLLM
- [Start vLLM Distribution](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/self_hosted_distro/remote-vllm.html)
:::

:::{tab-item} tgi
- [Start TGI Distribution](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/self_hosted_distro/tgi.html)
:::

:::{tab-item} ollama
- [Start Ollama Distribution](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/self_hosted_distro/ollama.html)
:::

:::{tab-item} together
- [Start Together Distribution](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/self_hosted_distro/together.html)
:::

:::{tab-item} fireworks
- [Start Fireworks Distribution](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/self_hosted_distro/fireworks.html)
:::

::::

##### Troubleshooting
- If you encounter any issues, search through our [GitHub Issues](https://github.com/meta-llama/llama-stack/issues), or file an new issue.
- Use `--port <PORT>` flag to use a different port number. For docker run, update the `-p <PORT>:<PORT>` flag.


## Step 2. Run Llama Stack App

### Chat Completion Test
Once the server is set up, we can test it with a client to verify it's working correctly. The following command will send a chat completion request to the server's `/inference/chat_completion` API:

```bash
$ curl http://localhost:5000/alpha/inference/chat-completion \
-H "Content-Type: application/json" \
-d '{
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
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
