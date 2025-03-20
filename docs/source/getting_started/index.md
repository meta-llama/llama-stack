# Quick Start

In this guide, we'll walk through how you can use the Llama Stack (server and client SDK) to test a simple RAG agent.

A Llama Stack agent is a simple integrated system that can perform tasks by combining a Llama model for reasoning with tools (e.g., RAG, web search, code execution, etc.) for taking actions.

In Llama Stack, we provide a server exposing multiple APIs. These APIs are backed by implementations from different providers. For this guide, we will use [Ollama](https://ollama.com/) as the inference provider.


### 1. Start Ollama

```bash
ollama run llama3.2:3b-instruct-fp16 --keepalive 60m
```

By default, Ollama keeps the model loaded in memory for 5 minutes which can be too short. We set the `--keepalive` flag to 60 minutes to ensure the model remains loaded for sometime.

```{admonition} Note
:class: tip

If you do not have ollama, you can install it from [here](https://ollama.com/download).
```


### 2. Pick a client environment

Llama Stack has a service-oriented architecture, so every interaction with the Stack happens through an REST interface. You can interact with the Stack in two ways:

* Install the `llama-stack-client` PyPI package and point `LlamaStackClient` to a local or remote Llama Stack server.
* Or, install the `llama-stack` PyPI package and use the Stack as a library using `LlamaStackAsLibraryClient`.

```{admonition} Note
:class: tip

The API is **exactly identical** for both clients.
```

:::{dropdown} Starting up the Llama Stack server
The Llama Stack server can be configured flexibly so you can mix-and-match various providers for its individual API components -- beyond Inference, these include Vector IO, Agents, Telemetry, Evals, Post Training, etc.

To get started quickly, we provide various container images for the server component that work with different inference providers out of the box. For this guide, we will use `llamastack/distribution-ollama` as the container image. If you'd like to build your own image or customize the configurations, please check out [this guide](../references/index.md).

Lets setup some environment variables that we will use in the rest of the guide.
```bash
export INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
export LLAMA_STACK_PORT=8321
```

Next you can create a local directory to mount into the container’s file system.
```bash
mkdir -p ~/.llama
```

Then you can start the server using the container tool of your choice.  For example, if you are running Docker you can use the following command:
```bash
docker run -it \
  --pull always \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  llamastack/distribution-ollama \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env OLLAMA_URL=http://host.docker.internal:11434
```

As another example, to start the container with Podman, you can do the same but replace `docker` at the start of the command with `podman`. If you are using `podman` older than `4.7.0`, please also replace `host.docker.internal` in the `OLLAMA_URL` with `host.containers.internal`.

Configuration for this is available at `distributions/ollama/run.yaml`.

```{admonition} Note
:class: note

Docker containers run in their own isolated network namespaces on Linux. To allow the container to communicate with services running on the host via `localhost`, you need `--network=host`. This makes the container use the host’s network directly so it can connect to Ollama running on `localhost:11434`.

Linux users having issues running the above command should instead try the following:
```bash
docker run -it \
  --pull always \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  --network=host \
  llamastack/distribution-ollama \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env OLLAMA_URL=http://localhost:11434
```

:::


:::{dropdown} Installing the Llama Stack client CLI and SDK

You can interact with the Llama Stack server using various client SDKs.  Note that you must be using Python 3.10 or newer. We will use the Python SDK which you can install via `conda` or `virtualenv`.

For `conda`:
```bash
yes | conda create -n stack-client python=3.10
conda activate stack-client
pip install llama-stack-client
```

For `virtualenv`:
```bash
python -m venv stack-client
source stack-client/bin/activate
pip install llama-stack-client
```

Let's use the `llama-stack-client` CLI to check the connectivity to the server.

```bash
$ llama-stack-client configure --endpoint http://localhost:$LLAMA_STACK_PORT
> Enter the API key (leave empty if no key is needed):
Done! You can now use the Llama Stack Client CLI with endpoint http://localhost:8321

$ llama-stack-client models list

Available Models

┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ model_type   ┃ identifier                           ┃ provider_resource_id         ┃ metadata  ┃ provider_id ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ llm          │ meta-llama/Llama-3.2-3B-Instruct     │ llama3.2:3b-instruct-fp16    │           │ ollama      │
└──────────────┴──────────────────────────────────────┴──────────────────────────────┴───────────┴─────────────┘

Total models: 1
```

You can test basic Llama inference completion using the CLI too.
```bash
llama-stack-client \
  inference chat-completion \
  --message "hello, what model are you?"
```
:::

&nbsp;

### 3. Run inference with Python SDK

Here is a simple example to perform chat completions using the SDK.
```python
import os
import sys


def create_http_client():
    from llama_stack_client import LlamaStackClient

    return LlamaStackClient(
        base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}"
    )


def create_library_client(template="ollama"):
    from llama_stack import LlamaStackAsLibraryClient

    client = LlamaStackAsLibraryClient(template)
    if not client.initialize():
        print("llama stack not built properly")
        sys.exit(1)
    return client


client = (
    create_library_client()
)  # or create_http_client() depending on the environment you picked

# List available models
models = client.models.list()
print("--- Available models: ---")
for m in models:
    print(f"- {m.identifier}")
print()

response = client.inference.chat_completion(
    model_id=os.environ["INFERENCE_MODEL"],
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about coding"},
    ],
)
print(response.completion_message.content)
```

To run the above example, put the code in a file called `inference.py`, ensure your `conda` or `virtualenv` environment is active, and run the following:
```bash
pip install llama_stack
llama stack build --template ollama --image-type <conda|venv>
python inference.py
```

### 4. Your first RAG agent

Here is an example of a simple RAG (Retrieval Augmented Generation) chatbot agent which can answer questions about TorchTune documentation.

```python
import os
import uuid
from termcolor import cprint

from llama_stack_client import Agent, AgentEventLogger, RAGDocument


def create_http_client():
    from llama_stack_client import LlamaStackClient

    return LlamaStackClient(
        base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}"
    )


def create_library_client(template="ollama"):
    from llama_stack import LlamaStackAsLibraryClient

    client = LlamaStackAsLibraryClient(template)
    client.initialize()
    return client


client = (
    create_library_client()
)  # or create_http_client() depending on the environment you picked

# Documents to be used for RAG
urls = ["chat.rst", "llama3.rst", "memory_optimizations.rst", "lora_finetune.rst"]
documents = [
    RAGDocument(
        document_id=f"num-{i}",
        content=f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}",
        mime_type="text/plain",
        metadata={},
    )
    for i, url in enumerate(urls)
]

vector_providers = [
    provider for provider in client.providers.list() if provider.api == "vector_io"
]
provider_id = vector_providers[0].provider_id  # Use the first available vector provider

# Register a vector database
vector_db_id = f"test-vector-db-{uuid.uuid4().hex}"
client.vector_dbs.register(
    vector_db_id=vector_db_id,
    provider_id=provider_id,
    embedding_model="all-MiniLM-L6-v2",
    embedding_dimension=384,
)

# Insert the documents into the vector database
client.tool_runtime.rag_tool.insert(
    documents=documents,
    vector_db_id=vector_db_id,
    chunk_size_in_tokens=512,
)

rag_agent = Agent(
    client,
    model=os.environ["INFERENCE_MODEL"],
    # Define instructions for the agent ( aka system prompt)
    instructions="You are a helpful assistant",
    enable_session_persistence=False,
    # Define tools available to the agent
    tools=[
        {
            "name": "builtin::rag/knowledge_search",
            "args": {
                "vector_db_ids": [vector_db_id],
            },
        }
    ],
)
session_id = rag_agent.create_session("test-session")

user_prompts = [
    "How to optimize memory usage in torchtune? use the knowledge_search tool to get information.",
]

# Run the agent loop by calling the `create_turn` method
for prompt in user_prompts:
    cprint(f"User> {prompt}", "green")
    response = rag_agent.create_turn(
        messages=[{"role": "user", "content": prompt}],
        session_id=session_id,
    )
    for log in AgentEventLogger().log(response):
        log.print()
```

To run the above example, put the code in a file called `rag.py`, ensure your `conda` or `virtualenv` environment is active, and run the following:
```bash
pip install llama_stack
llama stack build --template ollama --image-type <conda|venv>
python rag.py
```

## Next Steps

- Learn more about Llama Stack [Concepts](../concepts/index.md)
- Learn how to [Build Llama Stacks](../distributions/index.md)
- See [References](../references/index.md) for more details about the llama CLI and Python SDK
- For example applications and more detailed tutorials, visit our [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repository.
