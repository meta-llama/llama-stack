# Llama Stack Quickstart Guide

This guide will walk you through setting up an end-to-end workflow with Llama Stack, enabling you to perform text generation using the `Llama3.2-3B-Instruct` model. Follow these steps to get started quickly.

If you're looking for more specific topics like tool calling or agent setup, we have a [Zero to Hero Guide](#next-steps) that covers everything from Tool Calling to Agents in detail. Feel free to skip to the end to explore the advanced topics you're interested in.

## Table of Contents
1. [Setup](#Setup)
2. [Build, Configure, and Run Llama Stack](#build-configure-and-run-llama-stack)
3. [Testing with `curl`](#testing-with-curl)
4. [Testing with Python](#testing-with-python)
5. [Next Steps](#next-steps)

---



## Setup

### 1. Prerequisite

Ensure you have the following installed on your system:

- **Conda**: A package, dependency, and environment management tool.


### 2. Installation
The `llama` CLI tool helps you manage the Llama Stack toolchain and agent systems. Follow these step to install

First activate and activate your conda environment
```
conda create --name my-env
conda activate my-env
```
Then install llama-stack with pip, you could also check out other installation methods [here](https://llama-stack.readthedocs.io/en/latest/cli_reference/index.html).

```bash
pip install llama-stack
```

After installation, the `llama` command should be available in your PATH.

### 3. Download Llama Models

Download the necessary Llama model checkpoints using the `llama` CLI:

```bash
llama download --model-id Llama3.2-3B-Instruct
```

Follow the CLI prompts to complete the download. You may need to accept a license agreement. Obtain an instant license [here](https://www.llama.com/llama-downloads/).

---

## Build, Configure, and Run Llama Stack

### 1. Build the Llama Stack Distribution

We will default to building the `meta-reference-gpu` distribution due to its optimized configuration tailored for inference tasks that utilize local GPU capabilities effectively. If you have limited GPU resources, prefer using a cloud-based instance or plan to run on a CPU, you can explore other distribution options [here](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html#decide-your-inference-provider).

```bash
llama stack build --template meta-reference-gpu --image-type conda
```


### 2. Run the Llama Stack Distribution
> Launching a distribution initializes and configures the necessary APIs and Providers, enabling seamless interaction with the underlying model.

Start the server with the configured stack:

```bash
cd llama-stack/distributions/meta-reference-gpu
llama stack run ./run.yaml
```

The server will start and listen on `http://localhost:5000` by default.

---

## Testing with `curl`

After setting up the server, verify it's working by sending a `POST` request using `curl`:

```bash
curl http://localhost:5000/inference/chat_completion \
-H "Content-Type: application/json" \
-d '{
    "model": "Llama3.2-3B-Instruct",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write me a 2-sentence poem about the moon"}
    ],
    "sampling_params": {"temperature": 0.7, "seed": 42, "max_tokens": 512}
}'
```

**Expected Output:**
```json
{
  "completion_message": {
    "role": "assistant",
    "content": "The moon glows softly in the midnight sky,\nA beacon of wonder, as it catches the eye.",
    "stop_reason": "out_of_tokens",
    "tool_calls": []
  },
  "logprobs": null
}
```

---

## Testing with Python

You can also interact with the Llama Stack server using a simple Python script. Below is an example:

### 1. Install Required Python Packages
The `llama-stack-client` library offers a robust and efficient python methods for interacting with the Llama Stack server.

```bash
pip install llama-stack-client
```

### 2. Create Python Script (`test_llama_stack.py`)
```bash
touch test_llama_stack.py
```

### 3. Create a Chat Completion Request in Python

```python
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import SystemMessage, UserMessage

# Initialize the client
client = LlamaStackClient(base_url="http://localhost:5000")

# Create a chat completion request
response = client.inference.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a two-sentence poem about llama."}
    ],
    model="Llama3.2-3B-Instruct",
)

# Print the response
print(response.completion_message.content)
```

### 4. Run the Python Script

```bash
python test_llama_stack.py
```

**Expected Output:**
```
The moon glows softly in the midnight sky,
A beacon of wonder, as it catches the eye.
```

With these steps, you should have a functional Llama Stack setup capable of generating text using the specified model. For more detailed information and advanced configurations, refer to some of our documentation below.

---

## Next Steps

**Explore Other Guides**: Dive deeper into specific topics by following these guides:
- [Understanding Distribution](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html#decide-your-inference-provider)
- [Inference 101](00_Inference101.ipynb)
- [Local and Cloud Model Toggling 101](00_Local_Cloud_Inference101.ipynb)
- [Prompt Engineering](01_Prompt_Engineering101.ipynb)
- [Chat with Image - LlamaStack Vision API](02_Image_Chat101.ipynb)
- [Tool Calling: How to and Details](03_Tool_Calling101.ipynb)
- [Memory API: Show Simple In-Memory Retrieval](04_Memory101.ipynb)
- [Using Safety API in Conversation](05_Safety101.ipynb)
- [Agents API: Explain Components](06_Agents101.ipynb)


**Explore Client SDKs**: Utilize our client SDKs for various languages to integrate Llama Stack into your applications:
  - [Python SDK](https://github.com/meta-llama/llama-stack-client-python)
  - [Node SDK](https://github.com/meta-llama/llama-stack-client-node)
  - [Swift SDK](https://github.com/meta-llama/llama-stack-client-swift)
  - [Kotlin SDK](https://github.com/meta-llama/llama-stack-client-kotlin)

**Advanced Configuration**: Learn how to customize your Llama Stack distribution by referring to the [Building a Llama Stack Distribution](./building_distro.md) guide.

**Explore Example Apps**: Check out [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) for example applications built using Llama Stack.


---
