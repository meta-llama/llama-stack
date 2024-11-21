# Ollama Quickstart Guide

This guide will walk you through setting up an end-to-end workflow with Llama Stack with ollama, enabling you to perform text generation using the `Llama3.2-1B-Instruct` model. Follow these steps to get started quickly.

If you're looking for more specific topics like tool calling or agent setup, we have a [Zero to Hero Guide](#next-steps) that covers everything from Tool Calling to Agents in detail. Feel free to skip to the end to explore the advanced topics you're interested in.

> If you'd prefer not to set up a local server, explore our notebook on [tool calling with the Together API](Tool_Calling101_Using_Together's_Llama_Stack_Server.ipynb). This guide will show you how to leverage Together.ai's Llama Stack Server API, allowing you to get started with Llama Stack without the need for a locally built and running server.

## Table of Contents
1. [Setup ollama](#setup-ollama)
2. [Install Dependencies and Set Up Environment](#install-dependencies-and-set-up-environment)
3. [Build, Configure, and Run Llama Stack](#build-configure-and-run-llama-stack)
4. [Run Ollama Model](#run-ollama-model)
5. [Next Steps](#next-steps)

---

## Setup ollama

1. **Download Ollama App**:
   - Go to [https://ollama.com/download](https://ollama.com/download).
   - Download and unzip `Ollama-darwin.zip`.
   - Run the `Ollama` application.

1. **Download the Ollama CLI**:
   - Ensure you have the `ollama` command line tool by downloading and installing it from the same website.

1. **Start ollama server**:
   - Open the terminal and run:
      ```
      ollama serve
      ```

1. **Run the model**:
   - Open the terminal and run:
     ```bash
     ollama run llama3.2:3b-instruct-fp16
     ```
     **Note**: The supported models for llama stack for now is listed in [here](https://github.com/meta-llama/llama-stack/blob/main/llama_stack/providers/remote/inference/ollama/ollama.py#L43)


---

## Install Dependencies and Set Up Environment

1. **Create a Conda Environment**:
   - Create a new Conda environment with Python 3.11:
     ```bash
     conda create -n hack python=3.11
     ```
   - Activate the environment:
     ```bash
     conda activate hack
     ```

2. **Install ChromaDB**:
   - Install `chromadb` using `pip`:
     ```bash
     pip install chromadb
     ```

3. **Run ChromaDB**:
   - Start the ChromaDB server:
     ```bash
     chroma run --host localhost --port 8000 --path ./my_chroma_data
     ```

4. **Install Llama Stack**:
   - Open a new terminal and install `llama-stack`:
     ```bash
     conda activate hack
     pip install llama-stack
     ```

---

## Build, Configure, and Run Llama Stack

1. **Build the Llama Stack**:
   - Build the Llama Stack using the `ollama` template:
     ```bash
     llama stack build --template ollama --image-type conda
     ```

2. **Edit Configuration**:
   - Modify the `ollama-run.yaml` file located at `/Users/yourusername/.llama/distributions/llamastack-ollama/ollama-run.yaml`:
     - Change the `chromadb` port to `8000`.
     - Remove the `pgvector` section if present.

3. **Run the Llama Stack**:
   - Run the stack with the configured YAML file:
     ```bash
     llama stack run /path/to/your/distro/llamastack-ollama/ollama-run.yaml --port 5050
     ```
     Note:
        1. Everytime you run a new model with `ollama run`, you will need to restart the llama stack. Otherwise it won't see the new model

The server will start and listen on `http://localhost:5050`.

---

## Testing with `curl`

After setting up the server, open a new terminal window and verify it's working by sending a `POST` request using `curl`:

```bash
curl http://localhost:5050/inference/chat_completion \
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

You can check the available models with the command `llama-stack-client models list`.

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

### 1. Active Conda Environment and Install Required Python Packages
The `llama-stack-client` library offers a robust and efficient python methods for interacting with the Llama Stack server.

```bash
conda activate your-llama-stack-conda-env
pip install llama-stack-client
```

### 2. Create Python Script (`test_llama_stack.py`)
```bash
touch test_llama_stack.py
```

### 3. Create a Chat Completion Request in Python

```python
from llama_stack_client import LlamaStackClient

# Initialize the client
client = LlamaStackClient(base_url="http://localhost:5050")

# Create a chat completion request
response = client.inference.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a two-sentence poem about llama."}
    ],
    model="llama3.2:1b",
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

This command initializes the model to interact with your local Llama Stack instance.

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