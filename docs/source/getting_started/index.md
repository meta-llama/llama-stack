# Getting Started with Llama Stack


In this guide, we'll walk through using ollama as the inference provider and build a simple python application that uses the Llama Stack Client SDK

Llama stack consists of a distribution server and an accompanying client SDK. The distribution server can be configured for different providers for inference, memory, agents, evals etc. This configuration is defined in a yaml file called `run.yaml`.

Running inference on the underlying Llama model is one of the most critical requirements. Depending on what hardware you have available, you have various options. Note that each option have different necessary prerequisites. We will use ollama as the inference provider as it is the easiest to get started with.

### Step 1. Start the inference server
```bash
export LLAMA_STACK_PORT=5001
export INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
# ollama names this model differently, and we must use the ollama name when loading the model
export OLLAMA_INFERENCE_MODEL="llama3.2:3b-instruct-fp16"
ollama run $OLLAMA_INFERENCE_MODEL --keepalive 60m
```

### Step 2. Start the Llama Stack server

```bash
export LLAMA_STACK_PORT=5001
docker run \
  -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  llamastack/distribution-ollama \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env OLLAMA_URL=http://host.docker.internal:11434

```

### Step 3. Use the Llama Stack client SDK
```bash
pip install llama-stack-client
```

We will use the `llama-stack-client` CLI to check the connectivity to the server. This should be installed in your environment if you installed the SDK.
```bash
llama-stack-client --endpoint http://localhost:5001 models list
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ identifier                       ┃ provider_id ┃ provider_resource_id      ┃ metadata ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ meta-llama/Llama-3.2-3B-Instruct │ ollama      │ llama3.2:3b-instruct-fp16 │ {}       │
└──────────────────────────────────┴─────────────┴───────────────────────────┴──────────┘
```

Chat completion using the CLI
```bash
llama-stack-client --endpoint http://localhost:5001 inference chat_completion --message "hello, what model are you?"
```

Simple python example using the client SDK
```python
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url="http://localhost:5001")

# List available models
models = client.models.list()
print(models)

# Simple chat completion
response = client.inference.chat_completion(
    model_id="meta-llama/Llama-3.2-3B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about coding"}
    ]
)
print(response.completion_message.content)
```

### Step 4. Your first RAG agent
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

import fire

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types import Attachment
from llama_stack_client.types.agent_create_params import AgentConfig


async def run_main(host: str, port: int, disable_safety: bool = False):
    urls = [
        "memory_optimizations.rst",
        "chat.rst",
        "llama3.rst",
        "datasets.rst",
        "qat_finetune.rst",
        "lora_finetune.rst",
    ]

    attachments = [
        Attachment(
            content=f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}",
            mime_type="text/plain",
        )
        for i, url in enumerate(urls)
    ]

    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
    )

    available_shields = [shield.identifier for shield in client.shields.list()]
    if not available_shields:
        print("No available shields. Disable safety.")
    else:
        print(f"Available shields found: {available_shields}")
    available_models = [model.identifier for model in client.models.list()]
    if not available_models:
        raise ValueError("No available models")
    else:
        selected_model = available_models[0]
        print(f"Using model: {selected_model}")

    agent_config = AgentConfig(
        model=selected_model,
        instructions="You are a helpful assistant",
        sampling_params={
            "strategy": "greedy",
            "temperature": 1.0,
            "top_p": 0.9,
        },
        tools=[
            {
                "type": "memory",
                "memory_bank_configs": [],
                "query_generator_config": {"type": "default", "sep": " "},
                "max_tokens_in_context": 4096,
                "max_chunks": 10,
            },
        ],
        tool_choice="auto",
        tool_prompt_format="json",
        input_shields=available_shields if available_shields else [],
        output_shields=available_shields if available_shields else [],
        enable_session_persistence=False,
    )

    agent = Agent(client, agent_config)
    session_id = agent.create_session("test-session")
    print(f"Created session_id={session_id} for Agent({agent.agent_id})")

    user_prompts = [
        (
            "I am attaching some documentation for Torchtune. Help me answer questions I will ask next.",
            attachments,
        ),
        (
            "What are the top 5 topics that were explained? Only list succinct bullet points.",
            None,
        ),
        (
            "Was anything related to 'Llama3' discussed, if so what?",
            None,
        ),
        (
            "Tell me how to use LoRA",
            None,
        ),
        (
            "What about Quantization?",
            None,
        ),
    ]

    for prompt in user_prompts:
        response = agent.create_turn(
            messages=[
                {
                    "role": "user",
                    "content": prompt[0],
                }
            ],
            attachments=prompt[1],
            session_id=session_id,
        )

        async for log in EventLogger().log(response):
            log.print()


def main(host: str, port: int):
    asyncio.run(run_main(host, port))


if __name__ == "__main__":
    fire.Fire(main)
```

## Next Steps

- You can mix and match different providers for inference, memory, agents, evals etc. See [Building custom distributions](../distributions/index.md)
- [Developer Cookbook](developer_cookbook.md)

For example applications and more detailed tutorials, visit our [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repository.
