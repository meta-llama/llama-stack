# Why Llama Stack?

Building production AI applications today requires solving multiple challenges:

**Infrastructure Complexity**
- Running large language models efficiently requires specialized infrastructure.
- Different deployment scenarios (local development, cloud, edge) need different solutions.
- Moving from development to production often requires significant rework.

**Essential Capabilities**
- Safety guardrails and content filtering are necessary in an enterprise setting.
- Just model inference is not enough - Knowledge retrieval and RAG capabilities are required.
- Nearly any application needs composable multi-step workflows.
- Finally, without monitoring, observability and evaluation, you end up operating in the dark.

**Lack of Flexibility and Choice**
- Directly integrating with multiple providers creates tight coupling.
- Different providers have different APIs and abstractions.
- Changing providers requires significant code changes.


### The Vision: A Universal Stack


```{image} ../../_static/llama-stack.png
:alt: Llama Stack
:width: 400px
```

Llama Stack defines and standardizes the core building blocks needed to bring generative AI applications to market. These building blocks are presented as interoperable APIs with a broad set of Service Providers providing their implementations.

#### Service-oriented Design
Unlike other frameworks, Llama Stack is built with a service-oriented, REST API-first approach. Such a design not only allows for seamless transitions from local to remote deployments but also forces the design to be more declarative. This restriction can result in a much simpler, robust developer experience. The same code works across different environments:

- Local development with CPU-only setups
- Self-hosted with GPU acceleration
- Cloud-hosted on providers like AWS, Fireworks, Together
- On-device for iOS and Android


#### Composability
The APIs we design are composable. An Agent abstractly depends on { Inference, Memory, Safety } APIs but does not care about the actual implementation details. Safety itself may require model inference and hence can depend on the Inference API.

#### Turnkey Solutions

We provide turnkey solutions for popular deployment scenarios. It should be easy to deploy a Llama Stack server on AWS or in a private data center. Either of these should allow a developer to get started with powerful agentic apps, model evaluations, or fine-tuning services in minutes.

We have built-in support for critical needs:

- Safety guardrails and content filtering
- Comprehensive evaluation capabilities
- Full observability and monitoring
- Provider federation and fallback

#### Focus on Llama Models
As a Meta-initiated project, we explicitly focus on Meta's Llama series of models. Supporting the broad set of open models is no easy task and we want to start with models we understand best.

#### Supporting the Ecosystem
There is a vibrant ecosystem of Providers which provide efficient inference or scalable vector stores or powerful observability solutions. We want to make sure it is easy for developers to pick and choose the best implementations for their use cases. We also want to make sure it is easy for new Providers to onboard and participate in the ecosystem.

Additionally, we have designed every element of the Stack such that APIs as well as Resources (like Models) can be federated.

#### Rich Provider Ecosystem

```{list-table}
:header-rows: 1

* - Provider
  - Local
  - Self-hosted
  - Cloud
* - Inference
  - Ollama
  - vLLM, TGI
  - Fireworks, Together, AWS
* - Memory
  - FAISS
  - Chroma, pgvector
  - Weaviate
* - Safety
  - Llama Guard
  - -
  - AWS Bedrock
```


### Unified API Layer

Llama Stack provides a consistent interface for:

- **Inference**: Run LLM models efficiently
- **Safety**: Apply content filtering and safety policies
- **Memory**: Store and retrieve knowledge for RAG
- **Agents**: Build multi-step workflows
- **Evaluation**: Test and improve application quality
