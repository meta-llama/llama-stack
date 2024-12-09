# Llama Stack

[![PyPI version](https://img.shields.io/pypi/v/llama_stack.svg)](https://pypi.org/project/llama_stack/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-stack)](https://pypi.org/project/llama-stack/)
[![Discord](https://img.shields.io/discord/1257833999603335178)](https://discord.gg/llama-stack)

[**Quick Start**](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html) | [**Documentation**](https://llama-stack.readthedocs.io/en/latest/index.html) | [**Zero-to-Hero Guide**](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide)

Llama Stack defines and standardizes the set of core building blocks needed to bring generative AI applications to market. These building blocks are presented in the form of interoperable APIs with a broad set of Service Providers providing their implementations.

<div style="text-align: center;">
  <img
    src="https://github.com/user-attachments/assets/33d9576d-95ea-468d-95e2-8fa233205a50"
    width="480"
    title="Llama Stack"
    alt="Llama Stack"
  />
</div>

Our goal is to provide pre-packaged implementations which can be operated in a variety of deployment environments: developers start iterating with Desktops or their mobile devices and can seamlessly transition to on-prem or public cloud deployments. At every point in this transition, the same set of APIs and the same developer experience is available.

> ⚠️ **Note**
> The Stack APIs are rapidly improving, but still very much work in progress and we invite feedback as well as direct contributions.


## APIs

We have working implementations of the following APIs today:
- Inference
- Safety
- Memory
- Agents
- Eval
- Telemetry

Alongside these APIs, we also related APIs for operating with associated resources (see [Concepts](https://llama-stack.readthedocs.io/en/latest/concepts/index.html#resources)):

- Models
- Shields
- Memory Banks
- EvalTasks
- Datasets
- Scoring Functions

We are also working on the following APIs which will be released soon:

- Post Training
- Synthetic Data Generation
- Reward Scoring

Each of the APIs themselves is a collection of REST endpoints.

## Philosophy

### Service-oriented design

Unlike other frameworks, Llama Stack is built with a service-oriented, REST API-first approach. Such a design not only allows for seamless transitions from a local to remote deployments, but also forces the design to be more declarative. We believe this restriction can result in a much simpler, robust developer experience. This will necessarily trade-off against expressivity however if we get the APIs right, it can lead to a very powerful platform.

### Composability

We expect the set of APIs we design to be composable. An Agent abstractly depends on { Inference, Memory, Safety } APIs but does not care about the actual implementation details. Safety itself may require model inference and hence can depend on the Inference API.

### Turnkey one-stop solutions

We expect to provide turnkey solutions for popular deployment scenarios. It should be easy to deploy a Llama Stack server on AWS or on a private data center. Either of these should allow a developer to get started with powerful agentic apps, model evaluations or fine-tuning services in a matter of minutes. They should all result in the same uniform observability and developer experience.

### Focus on Llama models

As a Meta initiated project, we have started by explicitly focusing on Meta's Llama series of models. Supporting the broad set of open models is no easy task and we want to start with models we understand best.

### Supporting the Ecosystem

There is a vibrant ecosystem of Providers which provide efficient inference or scalable vector stores or powerful observability solutions. We want to make sure it is easy for developers to pick and choose the best implementations for their use cases. We also want to make sure it is easy for new Providers to onboard and participate in the ecosystem.

Additionally, we have designed every element of the Stack such that APIs as well as Resources (like Models) can be federated.


## Supported Llama Stack Implementations
### API Providers
|  **API Provider Builder** |  **Environments** | **Agents** | **Inference** | **Memory** | **Safety** | **Telemetry** |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|  Meta Reference  |  Single Node | :heavy_check_mark:  |  :heavy_check_mark:  |  :heavy_check_mark:  |  :heavy_check_mark:  |  :heavy_check_mark:  |
|  Cerebras  |  Hosted  |   | :heavy_check_mark:  |    |    |   |
|  Fireworks  |  Hosted  | :heavy_check_mark:  | :heavy_check_mark:  |  :heavy_check_mark:  |    |   |
|  AWS Bedrock  |  Hosted  |    |  :heavy_check_mark:  |    | :heavy_check_mark:  | |
|  Together  |  Hosted  |  :heavy_check_mark:  |  :heavy_check_mark:  |   | :heavy_check_mark:  |  |
|  Ollama  | Single Node   |    |  :heavy_check_mark:  |    |   |
|  TGI  |  Hosted and Single Node  |    |  :heavy_check_mark:  |    |   |
| Chroma | Single Node |  |  | :heavy_check_mark: |  |  |
| PG Vector | Single Node |  |  | :heavy_check_mark: |  |  |
| PyTorch ExecuTorch | On-device iOS | :heavy_check_mark:  | :heavy_check_mark:  |  |  |
|  Nutanix AI  |  Hosted  |  | :heavy_check_mark:  |  |    |   |

### Distributions

| **Distribution** 	|           **Llama Stack Docker**           	| Start This Distribution 	|
|:----------------:	|:------------------------------------------:	|:-----------------------:	|
|  Meta Reference  	| [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general) 	|       [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html)       	|
|  Meta Reference Quantized  	| [llamastack/distribution-meta-reference-quantized-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-quantized-gpu/general) 	|       [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-quantized-gpu.html)       	|
|      Cerebras     |       [llamastack/distribution-cerebras](https://hub.docker.com/repository/docker/llamastack/distribution-cerebras/general)       	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/self_hosted_distro/cerebras.html)       	|
|      Ollama      	|       [llamastack/distribution-ollama](https://hub.docker.com/repository/docker/llamastack/distribution-ollama/general)       	|       [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/ollama.html)       	|
|        TGI       	|         [llamastack/distribution-tgi](https://hub.docker.com/repository/docker/llamastack/distribution-tgi/general)        	|       [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/tgi.html)       	|
|        Together       	|         [llamastack/distribution-together](https://hub.docker.com/repository/docker/llamastack/distribution-together/general)        	|       [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/together.html)       	|
|        Fireworks       	|         [llamastack/distribution-fireworks](https://hub.docker.com/repository/docker/llamastack/distribution-fireworks/general)        	|       [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/fireworks.html)       	|
|        Nutanix       	|         [distribution-nutanix](https://hub.docker.com/repository/docker/jinanz/distribution-nutanix/general)        	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/remote_hosted_distro/nutanix.html)       	|

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
    $CONDA_PREFIX/bin/pip install -e .
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
