# NVIDIA Inference Provider for LlamaStack

This provider enables running inference using NVIDIA NIM.

## Features
- Endpoints for completions, chat completions, and embeddings for registered models

## Getting Started

### Prerequisites

- LlamaStack with NVIDIA configuration
- Access to NVIDIA NIM deployment
- NIM for model to use for inference is deployed

### Setup

Build the NVIDIA environment:

```bash
llama stack build --distro nvidia --image-type venv
```

### Basic Usage using the LlamaStack Python Client

#### Initialize the client

```python
import os

os.environ["NVIDIA_API_KEY"] = (
    ""  # Required if using hosted NIM endpoint. If self-hosted, not required.
)
os.environ["NVIDIA_BASE_URL"] = "http://nim.test"  # NIM URL

from llama_stack.core.library_client import LlamaStackAsLibraryClient

client = LlamaStackAsLibraryClient("nvidia")
client.initialize()
```

### Create Completion

```python
response = client.completion(
    model_id="meta-llama/Llama-3.1-8b-Instruct",
    content="Complete the sentence using one word: Roses are red, violets are :",
    stream=False,
    sampling_params={
        "max_tokens": 50,
    },
)
print(f"Response: {response.content}")
```

### Create Chat Completion

```python
response = client.chat_completion(
    model_id="meta-llama/Llama-3.1-8b-Instruct",
    messages=[
        {
            "role": "system",
            "content": "You must respond to each message with only one word",
        },
        {
            "role": "user",
            "content": "Complete the sentence using one word: Roses are red, violets are:",
        },
    ],
    stream=False,
    sampling_params={
        "max_tokens": 50,
    },
)
print(f"Response: {response.completion_message.content}")
```

### Create Embeddings
```python
response = client.embeddings(
    model_id="meta-llama/Llama-3.1-8b-Instruct", contents=["foo", "bar", "baz"]
)
print(f"Embeddings: {response.embeddings}")
```
