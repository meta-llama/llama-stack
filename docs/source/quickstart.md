
# Quickstart

This guide will walk you through the steps to set up an end-to-end workflow with Llama Stack. It focuses on building a Llama Stack distribution and starting up a Llama Stack server. See our [documentation](../README.md) for more on Llama Stack's capabilities, or visit [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main) for example apps.


## 0. Prerequsite
Feel free to skip this step if you already have the prerequsite installed.

1. conda (steps to install)
2.


## 1. Installation

The `llama` CLI tool helps you manage the Llama toolchain & agentic systems. After installing the `llama-stack` package, the `llama` command should be available in your path.

**Install as a package**:
   Install directly from [PyPI](https://pypi.org/project/llama-stack/) with:
   ```bash
   pip install llama-stack
   ```

## 2. Download Llama models:


   ```
   llama download --model-id Llama3.1-8B-Instruct
   ```
   You will have to follow the instructions in the cli to complete the download, get a instant license here: URL to license.

## 3. Build->Configure->Run via Conda:
   For development, build a LlamaStack distribution from scratch.

   **`llama stack build`**
   Enter build information interactively:
   ```bash
   llama stack build
   ```

   **`llama stack configure`**
   Run `llama stack configure <name>` using the name from the build step.
   ```bash
   llama stack configure my-local-stack
   ```

   **`llama stack run`**
   Start the server with:
   ```bash
   llama stack run my-local-stack
   ```

## 4. Testing with Client

After setup, test the server with a POST request:
```bash
curl http://localhost:5000/inference/chat_completion \
-H "Content-Type: application/json" \
-d '{
    "model": "Llama3.1-8B-Instruct",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write me a 2-sentence poem about the moon"}
    ],
    "sampling_params": {"temperature": 0.7, "seed": 42, "max_tokens": 512}
}'
```


## 5. Inference

After setup, test the server with a POST request:
```bash
curl http://localhost:5000/inference/chat_completion \
-H "Content-Type: application/json" \
-d '{
    "model": "Llama3.1-8B-Instruct",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write me a 2-sentence poem about the moon"}
    ],
    "sampling_params": {"temperature": 0.7, "seed": 42, "max_tokens": 512}
}'
```



Check our client SDKs for various languages: [Python](https://github.com/meta-llama/llama-stack-client-python), [Node](https://github.com/meta-llama/llama-stack-client-node), [Swift](https://github.com/meta-llama/llama-stack-client-swift), and [Kotlin](https://github.com/meta-llama/llama-stack-client-kotlin).

## Advanced Guides

For more on custom Llama Stack distributions, refer to our [Building a Llama Stack Distribution](./building_distro.md) guide.


## Next Steps:
check out

1.
2.
