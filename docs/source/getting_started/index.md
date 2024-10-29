# Getting Started with Llama Stack

At the end of the guide, you will have learnt how to:
- get a Llama Stack server up and running
- get a agent (with tool-calling, vector stores) which works with the above server

To see more example apps built using Llama Stack, see [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main).

## Starting Up Llama Stack Server

### Decide your
There are two ways to start a Llama Stack:

- **Docker**: we provide a number of pre-built Docker containers allowing you to get started instantly. If you are focused on application development, we recommend this option.
- **Conda**: the `llama` CLI provides a simple set of commands to build, configure and run a Llama Stack server containing the exact combination of providers you wish. We have provided various templates to make getting started easier.

Both of these provide options to run model inference using our reference implementations, Ollama, TGI, vLLM or even remote providers like Fireworks, Together, Bedrock, etc.

### Decide Your Inference Provider

Running inference of the underlying Llama model is one of the most critical requirements. Depending on what hardware you have available, you have various options:

- **Do you have access to a machine with powerful GPUs?**
If so, we suggest:
  - `distribution-meta-reference-gpu`:
    - [Docker]()
    - [Conda]()
  - `distribution-tgi`:
    - [Docker]()
    - [Conda]()

- **Are you running on a "regular" desktop machine?**
If so, we suggest:
  - `distribution-ollama`:
    - [Docker]()
    - [Conda]()

- **Do you have access to a remote inference provider like Fireworks, Togther, etc.?** If so, we suggest:
  - `distribution-fireworks`:
    - [Docker]()
    - [Conda]()
  - `distribution-together`:
    - [Docker]()
    - [Conda]()

## Testing with client
Once the server is setup, we can test it with a client to see the example outputs by . This will run the chat completion client and query the distribution’s `/inference/chat_completion` API. Send a POST request to the server:

```
curl http://localhost:5000/inference/chat_completion \
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

Check out our client SDKs for connecting to Llama Stack server in your preferred language, you can choose from [python](https://github.com/meta-llama/llama-stack-client-python), [node](https://github.com/meta-llama/llama-stack-client-node), [swift](https://github.com/meta-llama/llama-stack-client-swift), and [kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) programming languages to quickly build your applications.

You can find more example scripts with client SDKs to talk with the Llama Stack server in our [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repo.


```{toctree}
:maxdepth: 2

developer_cookbook
distributions/index
```
