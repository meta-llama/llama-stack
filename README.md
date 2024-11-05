<img src="https://github.com/user-attachments/assets/2fedfe0f-6df7-4441-98b2-87a1fd95ee1c" width="300" title="Llama Stack Logo" alt="Llama Stack Logo"/>

# Llama Stack

[![PyPI version](https://img.shields.io/pypi/v/llama_stack.svg)](https://pypi.org/project/llama_stack/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-stack)](https://pypi.org/project/llama-stack/)
[![Discord](https://img.shields.io/discord/1257833999603335178)](https://discord.gg/llama-stack)

[**Get Started**](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html) | [**Documentation**](https://llama-stack.readthedocs.io/en/latest/index.html)

This repository contains the Llama Stack API specifications as well as API Providers and Llama Stack Distributions.

The Llama Stack defines and standardizes the building blocks needed to bring generative AI applications to market. These blocks span the entire development lifecycle: from model training and fine-tuning, through product evaluation, to building and running AI agents in production. Beyond definition, we are building providers for the Llama Stack APIs. These were developing open-source versions and partnering with providers, ensuring developers can assemble AI solutions using consistent, interlocking pieces across platforms. The ultimate goal is to accelerate innovation in the AI space.

The Stack APIs are rapidly improving, but still very much work in progress and we invite feedback as well as direct contributions.


## APIs

The Llama Stack consists of the following set of APIs:

- Inference
- Safety
- Memory
- Agentic System
- Evaluation
- Post Training
- Synthetic Data Generation
- Reward Scoring

Each of the APIs themselves is a collection of REST endpoints.


## API Providers

A Provider is what makes the API real -- they provide the actual implementation backing the API.

As an example, for Inference, we could have the implementation be backed by open source libraries like `[ torch | vLLM | TensorRT ]` as possible options.

A provider can also be just a pointer to a remote REST service -- for example, cloud providers or dedicated inference providers could serve these APIs.


## Llama Stack Distribution

A Distribution is where APIs and Providers are assembled together to provide a consistent whole to the end application developer. You can mix-and-match providers -- some could be backed by local code and some could be remote. As a hobbyist, you can serve a small model locally, but can choose a cloud provider for a large model. Regardless, the higher level APIs your app needs to work with don't need to change at all. You can even imagine moving across the server / mobile-device boundary as well always using the same uniform set of APIs for developing Generative AI applications.

## Supported Llama Stack Implementations
### API Providers
|  **API Provider Builder** |  **Environments** | **Agents** | **Inference** | **Memory** | **Safety** | **Telemetry** |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|  Meta Reference  |  Single Node | :heavy_check_mark:  |  :heavy_check_mark:  |  :heavy_check_mark:  |  :heavy_check_mark:  |  :heavy_check_mark:  |
|  Fireworks  |  Hosted  | :heavy_check_mark:  | :heavy_check_mark:  |  :heavy_check_mark:  |    |   |
|  AWS Bedrock  |  Hosted  |    |  :heavy_check_mark:  |    | :heavy_check_mark:  | |
|  Together  |  Hosted  |  :heavy_check_mark:  |  :heavy_check_mark:  |   | :heavy_check_mark:  |  |
|  Ollama  | Single Node   |    |  :heavy_check_mark:  |    |   |
|  TGI  |  Hosted and Single Node  |    |  :heavy_check_mark:  |    |   |
| Chroma | Single Node |  |  | :heavy_check_mark: |  |  |
| PG Vector | Single Node |  |  | :heavy_check_mark: |  |  |
| PyTorch ExecuTorch | On-device iOS | :heavy_check_mark:  | :heavy_check_mark:  |  |  |

### Distributions

| **Distribution** 	|           **Llama Stack Docker**           	| Start This Distribution 	|    **Inference**   	|     **Agents**     	|     **Memory**     	|     **Safety**     	|    **Telemetry**   	|
|:----------------:	|:------------------------------------------:	|:-----------------------:	|:------------------:	|:------------------:	|:------------------:	|:------------------:	|:------------------:	|
|  Meta Reference  	| [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general) 	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/self_hosted_distro/meta-reference-gpu.html)       	| meta-reference 	| meta-reference 	| meta-reference; remote::pgvector; remote::chromadb	| meta-reference 	| meta-reference	|
|  Meta Reference Quantized  	| [llamastack/distribution-meta-reference-quantized-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-quantized-gpu/general) 	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/self_hosted_distro/meta-reference-quantized-gpu.html)       	| meta-reference-quantized 	| meta-reference 	| meta-reference; remote::pgvector; remote::chromadb	| meta-reference 	| meta-reference	|
|      Ollama      	|       [llamastack/distribution-ollama](https://hub.docker.com/repository/docker/llamastack/distribution-ollama/general)       	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/self_hosted_distro/ollama.html)       	| remote::ollama	| meta-reference 	| remote::pgvector; remote::chromadb 	|  meta-reference 	| meta-reference 	|
|        TGI       	|         [llamastack/distribution-tgi](https://hub.docker.com/repository/docker/llamastack/distribution-tgi/general)        	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/self_hosted_distro/tgi.html)       	| remote::tgi	| meta-reference 	| meta-reference; remote::pgvector; remote::chromadb 	| meta-reference 	| meta-reference 	|
|        Together       	|         [llamastack/distribution-together](https://hub.docker.com/repository/docker/llamastack/distribution-together/general)        	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/remote_hosted_distro/together.html)       	| remote::together 	| meta-reference | remote::weaviate | meta-reference 	| meta-reference  	|
|        Fireworks       	|         [llamastack/distribution-fireworks](https://hub.docker.com/repository/docker/llamastack/distribution-fireworks/general)        	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/remote_hosted_distro/fireworks.html)       	| remote::fireworks 	| meta-reference | remote::weaviate | meta-reference 	| meta-reference  	|
## Installation

You have two ways to install this repository:

1. **Install as a package**:
   You can install the repository directly from [PyPI](https://pypi.org/project/llama-stack/) by running the following command:
   ```bash
   pip install llama-stack
   ```

2. **Install from source**:
   If you prefer to install from the source code, follow these steps:
   ```bash
    mkdir -p ~/local
    cd ~/local
    git clone git@github.com:meta-llama/llama-stack.git

    conda create -n stack python=3.10
    conda activate stack

    cd llama-stack
    $CONDA_PREFIX/bin/pip install -e .
   ```

## Documentations

Please checkout our [Documentations](https://llama-stack.readthedocs.io/en/latest/index.html) page for more details.

* [CLI reference](https://llama-stack.readthedocs.io/en/latest/cli_reference/index.html)
    * Guide using `llama` CLI to work with Llama models (download, study prompts), and building/starting a Llama Stack distribution.
* [Getting Started](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)
    * Quick guide to start a Llama Stack server.
    * [Jupyter notebook](./docs/getting_started.ipynb) to walk-through how to use simple text and vision inference llama_stack_client APIs
* [Contributing](CONTRIBUTING.md)
    * [Adding a new API Provider](https://llama-stack.readthedocs.io/en/latest/api_providers/new_api_provider.html) to walk-through how to add a new API provider.

## Llama Stack Client SDK

|  **Language** |  **Client SDK** | **Package** |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift)
| Node   | [llama-stack-client-node](https://github.com/meta-llama/llama-stack-client-node) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) |

Check out our client SDKs for connecting to Llama Stack server in your preferred language, you can choose from [python](https://github.com/meta-llama/llama-stack-client-python), [node](https://github.com/meta-llama/llama-stack-client-node), [swift](https://github.com/meta-llama/llama-stack-client-swift), and [kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) programming languages to quickly build your applications.

You can find more example scripts with client SDKs to talk with the Llama Stack server in our [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repo.
