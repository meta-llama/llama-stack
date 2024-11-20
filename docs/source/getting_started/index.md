# Getting Started with Llama Stack

```{toctree}
:maxdepth: 2
:hidden:
```

In this guide, we'll walk through using ollama as the inference provider and build a simple python application that uses the Llama Stack Client SDK

Llama stack consists of a distribution server and an accompanying client SDK. The distribution server can be configured for different providers for inference, memory, agents, evals etc. This configuration is defined in a yaml file called `run.yaml`.

### Step 1. Start the ollama server
```bash
export LLAMA_STACK_PORT=5001
export INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
# ollama names this model differently, and we must use the ollama name when loading the model
export OLLAMA_INFERENCE_MODEL="llama3.2:3b-instruct-fp16"
ollama run $OLLAMA_INFERENCE_MODEL --keepalive 60m
```

### Step 2. Start the llama stack server

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

### Step 3. Install the client
```bash
pip install llama-stack-client
```

#### Check the connectivity to the server
```bash
llama-stack-client --endpoint http://localhost:5001 models list
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ identifier                       ┃ provider_id ┃ provider_resource_id      ┃ metadata ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ meta-llama/Llama-3.2-3B-Instruct │ ollama      │ llama3.2:3b-instruct-fp16 │ {}       │
└──────────────────────────────────┴─────────────┴───────────────────────────┴──────────┘
```

### Step 4. Sample app code
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

## Next Steps

For more advanced topics, check out:

- You can mix and match different providers for inference, memory, agents, evals etc. See [Building custom distributions](../distributions/index.md)
- [Developer Cookbook](developer_cookbook.md)

For example applications and more detailed tutorials, visit our [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repository.
