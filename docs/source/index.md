# Llama Stack

Llama Stack defines and standardizes the set of core building blocks needed to bring generative AI applications to market. These building blocks are presented in the form of interoperable APIs with a broad set of Service Providers providing their implementations.

```{image} ../_static/llama-stack.png
:alt: Llama Stack
:width: 400px
```

Our goal is to provide pre-packaged implementations which can be operated in a variety of deployment environments: developers start iterating with Desktops or their mobile devices and can seamlessly transition to on-prem or public cloud deployments. At every point in this transition, the same set of APIs and the same developer experience is available.

```{note}
The Stack APIs are rapidly improving but still a work-in-progress. We invite feedback as well as direct contributions.
```

## Quick Links

- New to Llama Stack? Start with the [Introduction](introduction/index) to understand our motivation and vision.
- Ready to build? Check out the [Quick Start](getting_started/index) to get started.
- Need specific providers? Browse [Distributions](distributions/index) to see all the options available.
- Want to contribute? See the [Contributing](contributing/index) guide.

## Available SDKs

We have a number of client-side SDKs available for different languages.

|  **Language** |  **Client SDK** | **Package** |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift)
| Node   | [llama-stack-client-node](https://github.com/meta-llama/llama-stack-client-node) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin)

## Supported Llama Stack Implementations

A number of "adapters" are available for some popular Inference and Memory (Vector Store) providers. For other APIs (particularly Safety and Agents), we provide *reference implementations* you can use to get started. We expect this list to grow over time. We are slowly onboarding more providers to the ecosystem as we get more confidence in the APIs.

|  **API Provider** |  **Environments** | **Agents** | **Inference** | **Memory** | **Safety** | **Telemetry** |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|  Meta Reference  |  Single Node | Y  |  Y  |  Y  |  Y  |  Y  |
|  Cerebras  |  Single Node  |   | Y  |    |    |   |
|  Fireworks  |  Hosted  | Y  | Y  |  Y  |    |   |
|  AWS Bedrock  |  Hosted  |    |  Y  |    | Y  | |
|  Together  |  Hosted  |  Y  |  Y  |   | Y  |  |
|  SambaNova  |  Hosted  |    |  Y  |   |   |  |
|  Ollama  | Single Node   |    |  Y  |    |   |
|  TGI  |  Hosted and Single Node  |    |  Y  |    |   |
|  [NVIDIA NIM](https://build.nvidia.com/nim?filters=nimType%3Anim_type_run_anywhere&q=llama)  |  Hosted and Single Node  |    |  Y  |    |   |
| Chroma | Single Node |  |  | Y |  |  |
| Postgres | Single Node |  |  | Y |  |  |
| PyTorch ExecuTorch | On-device iOS | Y  | Y  |  |  |
| PyTorch ExecuTorch | On-device Android |  | Y  |  |  |

```{toctree}
:hidden:
:maxdepth: 3

introduction/index
getting_started/index
concepts/index
distributions/index
building_applications/index
benchmark_evaluations/index
playground/index
contributing/index
references/index
```
