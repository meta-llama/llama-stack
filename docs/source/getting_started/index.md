# Quick Start

In this guide, we'll through how you can use the Llama Stack client SDK to build a simple RAG agent.

The most critical requirement for running the agent is running inference on the underlying Llama model. Depending on what hardware (GPUs) you have available, you have various options. We will use `Ollama` for this purpose as it is the easiest to get started with and yet robust.

First, let's set up some environment variables that we will use in the rest of the guide. Note that if you open up a new terminal, you will need to set these again.

```bash
export INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
# ollama names this model differently, and we must use the ollama name when loading the model
export OLLAMA_INFERENCE_MODEL="llama3.2:3b-instruct-fp16"
export LLAMA_STACK_PORT=5001
```

### 1. Start Ollama

```bash
ollama run $OLLAMA_INFERENCE_MODEL --keepalive 60m
```

By default, Ollama keeps the model loaded in memory for 5 minutes which can be too short. We set the `--keepalive` flag to 60 minutes to enspagents/agenure the model remains loaded for sometime.


### 2. Start the Llama Stack server

Llama Stack is based on a client-server architecture. It consists of a server which can be configured very flexibly so you can mix-and-match various providers for its individual API components -- beyond Inference, these include Memory, Agents, Telemetry, Evals and so forth.

```bash
docker run \
  -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  llamastack/distribution-ollama \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env OLLAMA_URL=http://host.docker.internal:11434
```

Configuration for this is available at `distributions/ollama/run.yaml`.


### 3. Use the Llama Stack client SDK

You can interact with the Llama Stack server using the `llama-stack-client` CLI or via the Python SDK.

```bash
pip install llama-stack-client
```

Let's use the `llama-stack-client` CLI to check the connectivity to the server.

```bash
llama-stack-client --endpoint http://localhost:$LLAMA_STACK_PORT models list
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ identifier                       ┃ provider_id ┃ provider_resource_id      ┃ metadata ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ meta-llama/Llama-3.2-3B-Instruct │ ollama      │ llama3.2:3b-instruct-fp16 │          │
└──────────────────────────────────┴─────────────┴───────────────────────────┴──────────┘
```

You can test basic Llama inference completion using the CLI too.
```bash
llama-stack-client --endpoint http://localhost:$LLAMA_STACK_PORT \
  inference chat_completion \
  --message "hello, what model are you?"
```

Here is a simple example to perform chat completions using Python instead of the CLI.
```python
import os
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}")

# List available models
models = client.models.list()
print(models)

response = client.inference.chat_completion(
    model_id=os.environ["INFERENCE_MODEL"],
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about coding"}
    ]
)
print(response.completion_message.content)
```

### 4. Your first RAG agent

Here is an example of a simple RAG agent that uses the Llama Stack client SDK.

```python
import asyncio
import os

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types import Attachment
from llama_stack_client.types.agent_create_params import AgentConfig


async def run_main():
    urls = ["chat.rst", "llama3.rst", "datasets.rst", "lora_finetune.rst"]
    attachments = [
        Attachment(
            content=f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}",
            mime_type="text/plain",
        )
        for i, url in enumerate(urls)
    ]

    client = LlamaStackClient(base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}")

    agent_config = AgentConfig(
        model=os.environ["INFERENCE_MODEL"],
        instructions="You are a helpful assistant",
        tools=[{"type": "memory"}],  # enable Memory aka RAG
    )

    agent = Agent(client, agent_config)
    session_id = agent.create_session("test-session")
    print(f"Created session_id={session_id} for Agent({agent.agent_id})")
    user_prompts = [
        (
            "I am attaching documentation for Torchtune. Help me answer questions I will ask next.",
            attachments,
        ),
        (
            "What are the top 5 topics that were explained? Only list succinct bullet points.",
            None,
        ),
    ]
    for prompt, attachments in user_prompts:
        response = agent.create_turn(
            messages=[{"role": "user", "content": prompt}],
            attachments=attachments,
            session_id=session_id,
        )
        async for log in EventLogger().log(response):
            log.print()


if __name__ == "__main__":
    asyncio.run(run_main())
```

## Next Steps

- Learn more about Llama Stack [Concepts](../concepts/index.md)
- Learn how to [Build Llama Stacks](../distributions/index.md)
- See [References](../references/index.md) for more details about the llama CLI and Python SDK
- For example applications and more detailed tutorials, visit our [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repository.
