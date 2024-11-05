# Llama Stack

Llama Stack defines and standardizes the building blocks needed to bring generative AI applications to market. It empowers developers building agentic applications by giving them options to operate in various environments (on-prem, cloud, single-node, on-device) while relying on a standard API interface and developer experience that's certified by Meta.

The Stack APIs are rapidly improving but still a work-in-progress. We invite feedback as well as direct contributions.


```{image} ../_static/llama-stack.png
:alt: Llama Stack
:width: 600px
:align: center
```

## APIs

The set of APIs in Llama Stack can be roughly split into two broad categories:

- APIs focused on Application development
  - Inference
  - Safety
  - Memory
  - Agentic System
  - Evaluation

- APIs focused on Model development
  - Evaluation
  - Post Training
  - Synthetic Data Generation
  - Reward Scoring

Each API is a collection of REST endpoints.

## API Providers

A Provider is what makes the API real – they provide the actual implementation backing the API.

As an example, for Inference, we could have the implementation be backed by open source libraries like [ torch | vLLM | TensorRT ] as possible options.

A provider can also be a relay to a remote REST service – ex. cloud providers or dedicated inference providers that serve these APIs.

## Distribution

A Distribution is where APIs and Providers are assembled together to provide a consistent whole to the end application developer. You can mix-and-match providers – some could be backed by local code and some could be remote. As a hobbyist, you can serve a small model locally, but can choose a cloud provider for a large model. Regardless, the higher level APIs your app needs to work with don't need to change at all. You can even imagine moving across the server / mobile-device boundary as well always using the same uniform set of APIs for developing Generative AI applications.

## Supported Llama Stack Implementations
### API Providers
|  **API Provider Builder** |  **Environments** | **Agents** | **Inference** | **Memory** | **Safety** | **Telemetry** |
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

### Distributions

| **Distribution** 	|           **Llama Stack Docker**           	| Start This Distribution 	|    **Inference**   	|     **Agents**     	|     **Memory**     	|     **Safety**     	|    **Telemetry**   	|
|:----------------:	|:------------------------------------------:	|:-----------------------:	|:------------------:	|:------------------:	|:------------------:	|:------------------:	|:------------------:	|
|  Meta Reference  	| [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general) 	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/self_hosted_distro/meta-reference-gpu.html)       	| meta-reference 	| meta-reference 	| meta-reference; remote::pgvector; remote::chromadb	| meta-reference 	| meta-reference	|
|  Meta Reference Quantized  	| [llamastack/distribution-meta-reference-quantized-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-quantized-gpu/general) 	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/self_hosted_distro/meta-reference-quantized-gpu.html)       	| meta-reference-quantized 	| meta-reference 	| meta-reference; remote::pgvector; remote::chromadb	| meta-reference 	| meta-reference	|
|      Ollama      	|       [llamastack/distribution-ollama](https://hub.docker.com/repository/docker/llamastack/distribution-ollama/general)       	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/self_hosted_distro/ollama.html)       	| remote::ollama	| meta-reference 	| remote::pgvector; remote::chromadb 	|  meta-reference 	| meta-reference 	|
|        TGI       	|         [llamastack/distribution-tgi](https://hub.docker.com/repository/docker/llamastack/distribution-tgi/general)        	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/self_hosted_distro/tgi.html)       	| remote::tgi	| meta-reference 	| meta-reference; remote::pgvector; remote::chromadb 	| meta-reference 	| meta-reference 	|
|        Together       	|         [llamastack/distribution-together](https://hub.docker.com/repository/docker/llamastack/distribution-together/general)        	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/remote_hosted_distro/together.html)       	| remote::together 	| meta-reference | remote::weaviate | meta-reference 	| meta-reference  	|
|        Fireworks       	|         [llamastack/distribution-fireworks](https://hub.docker.com/repository/docker/llamastack/distribution-fireworks/general)        	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/remote_hosted_distro/fireworks.html)       	| remote::fireworks 	| meta-reference | remote::weaviate | meta-reference 	| meta-reference  	|

## Llama Stack Client SDK

|  **Language** |  **Client SDK** | **Package** |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift)
| Node   | [llama-stack-client-node](https://github.com/meta-llama/llama-stack-client-node) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) |

Check out our client SDKs for connecting to Llama Stack server in your preferred language, you can choose from [python](https://github.com/meta-llama/llama-stack-client-python), [node](https://github.com/meta-llama/llama-stack-client-node), [swift](https://github.com/meta-llama/llama-stack-client-swift), and [kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) programming languages to quickly build your applications.

You can find more example scripts with client SDKs to talk with the Llama Stack server in our [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repo.


```{toctree}
:hidden:
:maxdepth: 3

getting_started/index
getting_started/ios_setup
cli_reference/index
cli_reference/download_models
api_providers/index
distribution_dev/index
```
