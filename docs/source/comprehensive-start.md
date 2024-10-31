
# Getting Started with Llama Stack

This guide will walk you through the steps to set up an end-to-end workflow with Llama Stack. It focuses on building a Llama Stack distribution and starting up a Llama Stack server. See our [documentation](../README.md) for more on Llama Stack's capabilities, or visit [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main) for example apps.

## Installation

The `llama` CLI tool helps you manage the Llama toolchain & agentic systems. After installing the `llama-stack` package, the `llama` command should be available in your path.

You can install this repository in two ways:

1. **Install as a package**:
   Install directly from [PyPI](https://pypi.org/project/llama-stack/) with:
   ```bash
   pip install llama-stack
   ```

2. **Install from source**:
   Follow these steps to install from the source code:
   ```bash
   mkdir -p ~/local
   cd ~/local
   git clone git@github.com:meta-llama/llama-stack.git

   conda create -n stack python=3.10
   conda activate stack

   cd llama-stack
   $CONDA_PREFIX/bin/pip install -e .
   ```

Refer to the [CLI Reference](./cli_reference.md) for details on Llama CLI commands.

## Starting Up Llama Stack Server

There are two ways to start the Llama Stack server:

1. **Using Docker**:
   We provide a pre-built Docker image of Llama Stack, available in the [distributions](../distributions/) folder.

   > **Note:** For GPU inference, set environment variables to specify the local directory with your model checkpoints and enable GPU inference.
   ```bash
   export LLAMA_CHECKPOINT_DIR=~/.llama
   ```
   Download Llama models with:
   ```
   llama download --model-id Llama3.1-8B-Instruct
   ```
   Start a Docker container with:
   ```bash
   cd llama-stack/distributions/meta-reference-gpu
   docker run -it -p 5000:5000 -v ~/.llama:/root/.llama -v ./run.yaml:/root/my-run.yaml --gpus=all distribution-meta-reference-gpu --yaml_config /root/my-run.yaml
   ```

   **Tip:** For remote providers, use `docker compose up` with scripts in the [distributions folder](../distributions/).

2. **Build->Configure->Run via Conda**:
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

## Testing with Client

After setup, test the server with a client:
```bash
cd /path/to/llama-stack
conda activate <env>

python -m llama_stack.apis.inference.client localhost 5000
```

You can also send a POST request:
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

For testing safety, run:
```bash
python -m llama_stack.apis.safety.client localhost 5000
```

Check our client SDKs for various languages: [Python](https://github.com/meta-llama/llama-stack-client-python), [Node](https://github.com/meta-llama/llama-stack-client-node), [Swift](https://github.com/meta-llama/llama-stack-client-swift), and [Kotlin](https://github.com/meta-llama/llama-stack-client-kotlin).

## Advanced Guides

For more on custom Llama Stack distributions, refer to our [Building a Llama Stack Distribution](./building_distro.md) guide.
