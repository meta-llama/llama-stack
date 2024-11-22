# Llama Stack

Llama Stack defines and standardizes the set of core building blocks needed to bring generative AI applications to market. These building blocks are presented in the form of interoperable APIs with a broad set of Service Providers providing their implementations. The APIs can be roughly split into two categories:

- APIs focused on Application development
  - Inference
  - Safety
  - Memory
  - Agents
  - Agent Evaluation

- APIs focused on Model development
  - Model Evaluation
  - Post Training
  - Synthetic Data Generation
  - Reward Scoring

Our goal is to provide pre-packaged implementations which can be operated in a variety of deployment environments: developers start iterating with Desktops or their mobile devices and can seamlessly transition to on-prem or public cloud deployments. At every point in this transition, the same set of APIs and the same developer experience is available.


```{image} ../_static/llama-stack.png
:alt: Llama Stack
:width: 400px
```

> [!NOTE]
> The Stack APIs are rapidly improving but still a work-in-progress. We invite feedback as well as direct contributions.

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

Llama Stack already has a number of "adapters" available for some popular Inference and Memory (Vector Store) providers. For other APIs (particularly Safety and Agents), we provide reference implementations you can use to get started. We expect this list to grow over time. We are slowly onboarding more providers to the ecosystem as we get more confidence in the APIs.

|  **API Provider** |  **Environments** | **Agents** | **Inference** | **Memory** | **Safety** | **Telemetry** |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|  Meta Reference  |  Single Node | Y  |  Y  |  Y  |  Y  |  Y  |
|  Fireworks  |  Hosted  | Y  | Y  |  Y  |    |   |
|  AWS Bedrock  |  Hosted  |    |  Y  |    | Y  | |
|  Together  |  Hosted  |  Y  |  Y  |   | Y  |  |
|  Ollama  | Single Node   |    |  Y  |    |   |
|  TGI  |  Hosted and Single Node  |    |  Y  |    |   |
| Chroma | Single Node |  |  | Y |  |  |
| PG Vector | Single Node |  |  | Y |  |  |
| PyTorch ExecuTorch | On-device iOS | Y  | Y  |  |  |

## Getting Started with "Distributions"

Distributions are pre-packaged (Docker) implementations of a specific set of Providers you can use to get started.

| **Distribution** 	|           **Llama Stack Docker**           	| Start This Distribution 	|
|:----------------:	|:------------------------------------------:	|:-----------------------:	|
|  Meta Reference  	| [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general) 	|       [Guide](distributions/self_hosted_distro/meta-reference-gpu.html)       	|
|  Meta Reference Quantized  	| [llamastack/distribution-meta-reference-quantized-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-quantized-gpu/general) 	|       [Guide](distributions/self_hosted_distro/meta-reference-quantized-gpu.html)       	|
|      Ollama      	|       [llamastack/distribution-ollama](https://hub.docker.com/repository/docker/llamastack/distribution-ollama/general)       	|       [Guide](distributions/self_hosted_distro/ollama.html)       	|
|        vLLM       	|         [llamastack/distribution-remote-vllm](https://hub.docker.com/repository/docker/llamastack/distribution-remote-vllm/general)        	|       [Guide](distributions/self_hosted_distro/vllm.html)       	|
|        TGI       	|         [llamastack/distribution-tgi](https://hub.docker.com/repository/docker/llamastack/distribution-tgi/general)        	|       [Guide](distributions/self_hosted_distro/tgi.html)       	|
|        Together       	|         [llamastack/distribution-together](https://hub.docker.com/repository/docker/llamastack/distribution-together/general)        	|       [Guide](distributions/remote_hosted_distro/together.html)       	|
|        Fireworks       	|         [llamastack/distribution-fireworks](https://hub.docker.com/repository/docker/llamastack/distribution-fireworks/general)        	|       [Guide](distributions/remote_hosted_distro/fireworks.html)       	|


## Llama Stack Client SDK

|  **Language** |  **Client SDK** | **Package** |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift)
| Node   | [llama-stack-client-node](https://github.com/meta-llama/llama-stack-client-node) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin)

Check out our client SDKs for connecting to Llama Stack server in your preferred language, you can choose from [python](https://github.com/meta-llama/llama-stack-client-python), [node](https://github.com/meta-llama/llama-stack-client-node), [swift](https://github.com/meta-llama/llama-stack-client-swift), and [kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) programming languages to quickly build your applications.

You can find more example scripts with client SDKs to talk with the Llama Stack server in our [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repo.


```{toctree}
:hidden:
:maxdepth: 3

getting_started/index
distributions/index
llama_cli_reference/index
llama_cli_reference/download_models
llama_stack_client_cli_reference/index
api_providers/index
distribution_dev/index
```
