# Llama Stack
Welcome to Llama Stack, the open-source framework for building generative AI applications.
```{admonition} Llama 4 is here!
:class: tip

Check out [Getting Started with Llama 4](https://colab.research.google.com/github/meta-llama/llama-stack/blob/main/docs/getting_started_llama4.ipynb)
```
```{admonition} News
:class: tip

Llama Stack {{ llama_stack_version }} is now available! See the {{ llama_stack_version_link }} for more details.
```


## What is Llama Stack?

Llama Stack defines and standardizes the core building blocks needed to bring generative AI applications to market. It provides a unified set of APIs with implementations from leading service providers, enabling seamless transitions between development and production environments. More specifically, it provides

- **Unified API layer** for Inference, RAG, Agents, Tools, Safety, Evals, and Telemetry.
- **Plugin architecture** to support the rich ecosystem of implementations of the different APIs in different environments like local development, on-premises, cloud, and mobile.
- **Prepackaged verified distributions** which offer a one-stop solution for developers to get started quickly and reliably in any environment
- **Multiple developer interfaces** like CLI and SDKs for Python, Node, iOS, and Android
- **Standalone applications** as examples for how to build production-grade AI applications with Llama Stack

```{image} ../_static/llama-stack.png
:alt: Llama Stack
:width: 400px
```

Our goal is to provide pre-packaged implementations (aka "distributions") which can be run in a variety of deployment environments. LlamaStack can assist you in your entire app development lifecycle - start iterating on local, mobile or desktop and seamlessly transition to on-prem or public cloud deployments. At every point in this transition, the same set of APIs and the same developer experience is available.

## How does Llama Stack work?
Llama Stack consists of a [server](./distributions/index.md) (with multiple pluggable API [providers](./providers/index.md)) and Client SDKs (see below) meant to
be used in your applications. The server can be run in a variety of environments, including local (inline)
development, on-premises, and cloud. The client SDKs are available for Python, Swift, Node, and
Kotlin.

## Quick Links

- Ready to build? Check out the [Quick Start](getting_started/index) to get started.
- Want to contribute? See the [Contributing](contributing/index) guide.

## Client SDKs

We have a number of client-side SDKs available for different languages.

|  **Language** |  **Client SDK** | **Package** |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift/tree/latest-release) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift)
| Node   | [llama-stack-client-node](https://github.com/meta-llama/llama-stack-client-node) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin/tree/latest-release) | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin)

## Supported Llama Stack Implementations

A number of "adapters" are available for some popular Inference and Vector Store providers. For other APIs (particularly Safety and Agents), we provide *reference implementations* you can use to get started. We expect this list to grow over time. We are slowly onboarding more providers to the ecosystem as we get more confidence in the APIs.

**Inference API**
|  **Provider** |  **Environments** |
| :----: | :----: |
|  Meta Reference  |  Single Node |
|  Ollama  | Single Node   |
|  Fireworks  |  Hosted  |
|  Together  |  Hosted  |
|  NVIDIA NIM  |  Hosted and Single Node  |
|  vLLM  | Hosted and Single Node |
|  TGI  |  Hosted and Single Node  |
|  AWS Bedrock  |  Hosted  |
|  Cerebras  |  Hosted  |
|  Groq  |  Hosted  |
|  SambaNova  |  Hosted  |
| PyTorch ExecuTorch | On-device iOS, Android |
|  OpenAI  |  Hosted  |
|  Anthropic  |  Hosted  |
|  Gemini  |  Hosted  |


**Vector IO API**
|  **Provider** |  **Environments** |
| :----: | :----: |
|  FAISS | Single Node |
|  SQLite-Vec| Single Node |
|  Chroma | Hosted and Single Node |
|  Milvus | Hosted and Single Node |
|  Postgres (PGVector) | Hosted and Single Node |
|  Weaviate | Hosted |

**Safety API**
|  **Provider** |  **Environments** |
| :----: | :----: |
|  Llama Guard | Depends on Inference Provider |
|  Prompt Guard | Single Node |
|  Code Scanner | Single Node |
|  AWS Bedrock | Hosted |


```{toctree}
:hidden:
:maxdepth: 3

self
getting_started/index
getting_started/detailed_tutorial
introduction/index
concepts/index
openai/index
providers/index
distributions/index
building_applications/index
playground/index
contributing/index
references/index
```
