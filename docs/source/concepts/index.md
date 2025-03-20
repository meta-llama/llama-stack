# Core Concepts


```{toctree}
:maxdepth: 1
:hidden:

evaluation_concepts
```

Given Llama Stack's service-oriented philosophy, a few concepts and workflows arise which may not feel completely natural in the LLM landscape, especially if you are coming with a background in other frameworks.


## APIs

A Llama Stack API is described as a collection of REST endpoints. We currently support the following APIs:

- **Inference**: run inference with a LLM
- **Safety**: apply safety policies to the output at a Systems (not only model) level
- **Agents**: run multi-step agentic workflows with LLMs with tool usage, memory (RAG), etc.
- **DatasetIO**: interface with datasets and data loaders
- **Scoring**: evaluate outputs of the system
- **Eval**: generate outputs (via Inference or Agents) and perform scoring
- **VectorIO**: perform operations on vector stores, such as adding documents, searching, and deleting documents
- **Telemetry**: collect telemetry data from the system

We are working on adding a few more APIs to complete the application lifecycle. These will include:
- **Batch Inference**: run inference on a dataset of inputs
- **Batch Agents**: run agents on a dataset of inputs
- **Post Training**: fine-tune a Llama model
- **Synthetic Data Generation**: generate synthetic data for model development

## API Providers

The goal of Llama Stack is to build an ecosystem where users can easily swap out different implementations for the same API. Examples for these include:
- LLM inference providers (e.g., Fireworks, Together, AWS Bedrock, Groq, Cerebras, SambaNova, vLLM, etc.),
- Vector databases (e.g., ChromaDB, Weaviate, Qdrant, Milvus, FAISS, PGVector, etc.),
- Safety providers (e.g., Meta's Llama Guard, AWS Bedrock Guardrails, etc.)

Providers come in two flavors:
- **Remote**: the provider runs as a separate service external to the Llama Stack codebase. Llama Stack contains a small amount of adapter code.
- **Inline**: the provider is fully specified and implemented within the Llama Stack codebase. It may be a simple wrapper around an existing library, or a full fledged implementation within Llama Stack.

Most importantly, Llama Stack always strives to provide at least one fully inline provider for each API so you can iterate on a fully featured environment locally.
## Resources

Some of these APIs are associated with a set of **Resources**. Here is the mapping of APIs to resources:

- **Inference**, **Eval** and **Post Training** are associated with `Model` resources.
- **Safety** is associated with `Shield` resources.
- **Tool Runtime** is associated with `ToolGroup` resources.
- **DatasetIO** is associated with `Dataset` resources.
- **VectorIO** is associated with `VectorDB` resources.
- **Scoring** is associated with `ScoringFunction` resources.
- **Eval** is associated with `Model` and `Benchmark` resources.

Furthermore, we allow these resources to be **federated** across multiple providers. For example, you may have some Llama models served by Fireworks while others are served by AWS Bedrock. Regardless, they will all work seamlessly with the same uniform Inference API provided by Llama Stack.

```{admonition} Registering Resources
:class: tip

Given this architecture, it is necessary for the Stack to know which provider to use for a given resource. This means you need to explicitly _register_ resources (including models) before you can use them with the associated APIs.
```

## Distributions

While there is a lot of flexibility to mix-and-match providers, often users will work with a specific set of providers (hardware support, contractual obligations, etc.) We therefore need to provide a _convenient shorthand_ for such collections. We call this shorthand a **Llama Stack Distribution** or a **Distro**. One can think of it as specific pre-packaged versions of the Llama Stack. Here are some examples:

**Remotely Hosted Distro**: These are the simplest to consume from a user perspective. You can simply obtain the API key for these providers, point to a URL and have _all_ Llama Stack APIs working out of the box. Currently, [Fireworks](https://fireworks.ai/) and [Together](https://together.xyz/) provide such easy-to-consume Llama Stack distributions.

**Locally Hosted Distro**: You may want to run Llama Stack on your own hardware. Typically though, you still need to use Inference via an external service. You can use providers like HuggingFace TGI, Fireworks, Together, etc. for this purpose. Or you may have access to GPUs and can run a [vLLM](https://github.com/vllm-project/vllm) or [NVIDIA NIM](https://build.nvidia.com/nim?filters=nimType%3Anim_type_run_anywhere&q=llama) instance. If you "just" have a regular desktop machine, you can use [Ollama](https://ollama.com/) for inference. To provide convenient quick access to these options, we provide a number of such pre-configured locally-hosted Distros.


**On-device Distro**: To run Llama Stack directly on an edge device (mobile phone or a tablet), we provide Distros for [iOS](https://llama-stack.readthedocs.io/en/latest/distributions/ondevice_distro/ios_sdk.html) and [Android](https://llama-stack.readthedocs.io/en/latest/distributions/ondevice_distro/android_sdk.html)
