# Llama Stack

[![PyPI version](https://img.shields.io/pypi/v/llama_stack.svg)](https://pypi.org/project/llama_stack/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-stack)](https://pypi.org/project/llama-stack/)
[![Discord](https://img.shields.io/discord/1257833999603335178)](https://discord.gg/llama-stack)

[**Quick Start**](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html) | [**Documentation**](https://llama-stack.readthedocs.io/en/latest/index.html) | [**Colab Notebook**](./docs/getting_started.ipynb)

Llama Stack defines and standardizes the core building blocks needed to bring generative AI applications to market. It provides a unified set of APIs with implementations from leading service providers, enabling seamless transitions between development and production environments.

We focus on making it easy to build production applications with the Llama model family - from the latest Llama 3.3 to specialized models like Llama Guard for safety.

<div style="text-align: center;">
  <img
    src="https://github.com/user-attachments/assets/33d9576d-95ea-468d-95e2-8fa233205a50"
    width="480"
    title="Llama Stack"
    alt="Llama Stack"
  />
</div>

## Key Features

- **Unified API Layer** for:
  - Inference: Run LLM models efficiently
  - Safety: Apply content filtering and safety policies
  - DatasetIO: Store and retrieve knowledge for RAG
  - Agents: Build multi-step agentic workflows
  - Evaluation: Test and improve model and agent quality
  - Telemetry: Collect and analyze usage data and complex agentic traces
  - Post Training ( Coming Soon ): Fine tune models for specific use cases

- **Rich Provider Ecosystem**
  - Local Development: Meta's Reference,Ollama, vLLM, TGI
  - Self-hosted: Chroma, pgvector, Nvidia NIM
  - Cloud: Fireworks, Together, Nvidia, AWS Bedrock, Groq, Cerebras
  - On-device: iOS and Android support

- **Built for Production**
  - Pre-packaged distributions for common deployment scenarios
  - Comprehensive evaluation capabilities
  - Full observability and monitoring
  - Provider federation and fallback


## Supported Llama Stack Implementations
### API Providers
|                                  **API Provider Builder**                                  |    **Environments**    |     **Agents**     |   **Inference**    |     **Memory**     |     **Safety**     |   **Telemetry**    |
|:------------------------------------------------------------------------------------------:|:----------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
|                                       Meta Reference                                       |      Single Node       | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|                                          Cerebras                                          |         Hosted         |                    | :heavy_check_mark: |                    |                    |                    |
|                                         Fireworks                                          |         Hosted         | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |                    |
|                                        AWS Bedrock                                         |         Hosted         |                    | :heavy_check_mark: |                    | :heavy_check_mark: |                    |
|                                          Together                                          |         Hosted         | :heavy_check_mark: | :heavy_check_mark: |                    | :heavy_check_mark: |                    |
|                                            Groq                                            |         Hosted         |                    | :heavy_check_mark: |                    |                    |                    |
|                                           Ollama                                           |      Single Node       |                    | :heavy_check_mark: |                    |                    |                    |
|                                            TGI                                             | Hosted and Single Node |                    | :heavy_check_mark: |                    |                    |                    |
| NVIDIA NIM | Hosted and Single Node |                    | :heavy_check_mark: |                    |                    |                    |
|                                           Chroma                                           |      Single Node       |                    |                    | :heavy_check_mark: |                    |                    |
|                                         PG Vector                                          |      Single Node       |                    |                    | :heavy_check_mark: |                    |                    |
|                                     PyTorch ExecuTorch                                     |     On-device iOS      | :heavy_check_mark: | :heavy_check_mark: |                    |                    |                    |
|                        vLLM                        | Hosted and Single Node |                    | :heavy_check_mark: |                    |                    |                    |

### Distributions

A Llama Stack Distribution (or "distro") is a pre-configured bundle of provider implementations for each API component. Distributions make it easy to get started with a specific deployment scenario - you can begin with a local development setup (eg. ollama) and seamlessly transition to production (eg. Fireworks) without changing your application code. Here are some of the distributions we support:

|               **Distribution**                |                                                                    **Llama Stack Docker**                                                                     |                                                 Start This Distribution                                                  |
|:---------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|
|                Meta Reference                 |           [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html)      |
|           Meta Reference Quantized            | [llamastack/distribution-meta-reference-quantized-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-quantized-gpu/general) | [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-quantized-gpu.html) |
|                   Cerebras                    |                     [llamastack/distribution-cerebras](https://hub.docker.com/repository/docker/llamastack/distribution-cerebras/general)                     |   [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/cerebras.html)   |
|                    Ollama                     |                       [llamastack/distribution-ollama](https://hub.docker.com/repository/docker/llamastack/distribution-ollama/general)                       |            [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/ollama.html)            |
|                      TGI                      |                          [llamastack/distribution-tgi](https://hub.docker.com/repository/docker/llamastack/distribution-tgi/general)                          |             [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/tgi.html)              |
|                   Together                    |                     [llamastack/distribution-together](https://hub.docker.com/repository/docker/llamastack/distribution-together/general)                     |           [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/together.html)           |
|                   Fireworks                   |                    [llamastack/distribution-fireworks](https://hub.docker.com/repository/docker/llamastack/distribution-fireworks/general)                    |          [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/fireworks.html)           |
| vLLM |                  [llamastack/distribution-remote-vllm](https://hub.docker.com/repository/docker/llamastack/distribution-remote-vllm/general)                  |         [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/remote-vllm.html)          |

## Installation

You have two ways to install this repository:

1. **Install as a package**:
   You can install the repository directly from [PyPI](https://pypi.org/project/llama-stack/) by running the following command:
   ```bash
   pip install llama-stack
   ```

2. **Install from source**:
   If you prefer to install from the source code, make sure you have [conda installed](https://docs.conda.io/projects/conda/en/stable).
   Then, follow these steps:
   ```bash
    mkdir -p ~/local
    cd ~/local
    git clone git@github.com:meta-llama/llama-stack.git

    conda create -n stack python=3.10
    conda activate stack

    cd llama-stack
    pip install -e .
   ```

## Documentation

Please checkout our [Documentation](https://llama-stack.readthedocs.io/en/latest/index.html) page for more details.

* [CLI reference](https://llama-stack.readthedocs.io/en/latest/references/llama_cli_reference/index.html)
    * Guide using `llama` CLI to work with Llama models (download, study prompts), and building/starting a Llama Stack distribution.
* [Getting Started](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)
    * Quick guide to start a Llama Stack server.
    * [Jupyter notebook](./docs/getting_started.ipynb) to walk-through how to use simple text and vision inference llama_stack_client APIs
    * The complete Llama Stack lesson [Colab notebook](https://colab.research.google.com/drive/1dtVmxotBsI4cGZQNsJRYPrLiDeT0Wnwt) of the new [Llama 3.2 course on Deeplearning.ai](https://learn.deeplearning.ai/courses/introducing-multimodal-llama-3-2/lesson/8/llama-stack).
    * A [Zero-to-Hero Guide](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide) that guide you through all the key components of llama stack with code samples.
* [Contributing](CONTRIBUTING.md)
    * [Adding a new API Provider](https://llama-stack.readthedocs.io/en/latest/contributing/new_api_provider.html) to walk-through how to add a new API provider.

## Llama Stack Client SDKs

|  **Language** |  **Client SDK** | **Package** |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift)
| Node   | [llama-stack-client-node](https://github.com/meta-llama/llama-stack-client-node) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin)

Check out our client SDKs for connecting to Llama Stack server in your preferred language, you can choose from [python](https://github.com/meta-llama/llama-stack-client-python), [node](https://github.com/meta-llama/llama-stack-client-node), [swift](https://github.com/meta-llama/llama-stack-client-swift), and [kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) programming languages to quickly build your applications.

You can find more example scripts with client SDKs to talk with the Llama Stack server in our [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repo.
