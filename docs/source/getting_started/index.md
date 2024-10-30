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

## Starting Up Llama Stack Server

### Decide Your Build Type
There are two ways to start a Llama Stack:

- **Docker**: we provide a number of pre-built Docker containers allowing you to get started instantly. If you are focused on application development, we recommend this option.
- **Conda**: the `llama` CLI provides a simple set of commands to build, configure and run a Llama Stack server containing the exact combination of providers you wish. We have provided various templates to make getting started easier.

Both of these provide options to run model inference using our reference implementations, Ollama, TGI, vLLM or even remote providers like Fireworks, Together, Bedrock, etc.

### Decide Your Inference Provider

Running inference of the underlying Llama model is one of the most critical requirements. Depending on what hardware you have available, you have various options:

- **Do you have access to a machine with powerful GPUs?**
If so, we suggest:
  - `distribution-meta-reference-gpu`:
    - [Docker](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/meta-reference-gpu.html#docker-start-the-distribution)
    - [Conda](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/meta-reference-gpu.html#docker-start-the-distribution)
  - `distribution-tgi`:
    - [Docker](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/tgi.html#docker-start-the-distribution-single-node-gpu)
    - [Conda](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/tgi.html#conda-tgi-server-llama-stack-run)

- **Are you running on a "regular" desktop machine?**
If so, we suggest:
  - `distribution-ollama`:
    - [Docker](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/ollama.html#docker-start-a-distribution-single-node-gpu)
    - [Conda](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/ollama.html#conda-ollama-run-llama-stack-run)

- **Do you have access to a remote inference provider like Fireworks, Togther, etc.?** If so, we suggest:
  - `distribution-together`:
    - [Docker](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/together.html#docker-start-the-distribution-single-node-cpu)
    - [Conda](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/together.html#conda-llama-stack-run-single-node-cpu)
  - `distribution-fireworks`:
    - [Docker](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/fireworks.html)
    - [Conda](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/fireworks.html#conda-llama-stack-run-single-node-cpu)


## Build Your Llama Stack App

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

::::{tab-set}

:::{tab-item} Label1
Content 1
:::

:::{tab-item} Label2
Content 2
:::

::::
