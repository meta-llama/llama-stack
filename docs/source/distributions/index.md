# Building Llama Stacks

```{toctree}
:maxdepth: 2
:hidden:

self_hosted_distro/index
remote_hosted_distro/index
ondevice_distro/index
```
## Introduction

Llama Stack Distributions are pre-built Docker containers/Conda environments that assemble APIs and Providers to provide a consistent whole to the end application developer.

These distributions allow you to mix-and-match providers - some could be backed by local code and some could be remote. This flexibility enables you to choose the optimal setup for your use case, such as serving a small model locally while using a cloud provider for larger models, all while maintaining a consistent API interface for your application.


## Decide Your Build Type
There are two ways to start a Llama Stack:

- **Docker**: we provide a number of pre-built Docker containers allowing you to get started instantly. If you are focused on application development, we recommend this option.
- **Conda**: the `llama` CLI provides a simple set of commands to build, configure and run a Llama Stack server containing the exact combination of providers you wish. We have provided various templates to make getting started easier.

Both of these provide options to run model inference using our reference implementations, Ollama, TGI, vLLM or even remote providers like Fireworks, Together, Bedrock, etc.

### Decide Your Inference Provider

Running inference on the underlying Llama model is one of the most critical requirements. Depending on what hardware you have available, you have various options. Note that each option have different necessary prerequisites.

- **Do you have access to a machine with powerful GPUs?**
If so, we suggest:
  - [distribution-meta-reference-gpu](./self_hosted_distro/meta-reference-gpu.md)
  - [distribution-tgi](./self_hosted_distro/tgi.md)

- **Are you running on a "regular" desktop machine?**
If so, we suggest:
  - [distribution-ollama](./self_hosted_distro/ollama.md)

- **Do you have an API key for a remote inference provider like Fireworks, Together, etc.?** If so, we suggest:
  - [distribution-together](./remote_hosted_distro/together.md)
  - [distribution-fireworks](./remote_hosted_distro/fireworks.md)

- **Do you want to run Llama Stack inference on your iOS / Android device** If so, we suggest:
  - [iOS](./ondevice_distro/ios_sdk.md)
  - [Android](https://github.com/meta-llama/llama-stack-client-kotlin) (coming soon)

Please see our pages in detail for the types of distributions we offer:

1. [Self-Hosted Distributions](./self_hosted_distro/index.md): If you want to run Llama Stack inference on your local machine.
2. [Remote-Hosted Distributions](./remote_hosted_distro/index.md): If you want to connect to a remote hosted inference provider.
3. [On-device Distributions](./ondevice_distro/index.md): If you want to run Llama Stack inference on your iOS / Android device.

## Building Your Own Distribution

### Prerequisites

```bash
$ git clone git@github.com:meta-llama/llama-stack.git
```


### Starting the Distribution

::::{tab-set}

:::{tab-item} meta-reference-gpu
##### System Requirements
Access to Single-Node GPU to start a local server.

##### Downloading Models
Please make sure you have Llama model checkpoints downloaded in `~/.llama` before proceeding. See [installation guide](../cli_reference/download_models.md) here to download the models.

```
$ ls ~/.llama/checkpoints
Llama3.1-8B           Llama3.2-11B-Vision-Instruct  Llama3.2-1B-Instruct  Llama3.2-90B-Vision-Instruct  Llama-Guard-3-8B
Llama3.1-8B-Instruct  Llama3.2-1B                   Llama3.2-3B-Instruct  Llama-Guard-3-1B              Prompt-Guard-86M
```

:::

:::{tab-item} vLLM
##### System Requirements
Access to Single-Node GPU to start a vLLM server.
:::

:::{tab-item} tgi
##### System Requirements
Access to Single-Node GPU to start a TGI server.
:::

:::{tab-item} ollama
##### System Requirements
Access to Single-Node CPU/GPU able to run ollama.
:::

:::{tab-item} together
##### System Requirements
Access to Single-Node CPU with Together hosted endpoint via API_KEY from [together.ai](https://api.together.xyz/signin).
:::

:::{tab-item} fireworks
##### System Requirements
Access to Single-Node CPU with Fireworks hosted endpoint via API_KEY from [fireworks.ai](https://fireworks.ai/).
:::

::::


::::{tab-set}
:::{tab-item} meta-reference-gpu
- [Start Meta Reference GPU Distribution](./self_hosted_distro/meta-reference-gpu.md)
:::

:::{tab-item} vLLM
- [Start vLLM Distribution](./self_hosted_distro/remote-vllm.md)
:::

:::{tab-item} tgi
- [Start TGI Distribution](./self_hosted_distro/tgi.md)
:::

:::{tab-item} ollama
- [Start Ollama Distribution](./self_hosted_distro/ollama.md)
:::

:::{tab-item} together
- [Start Together Distribution](./self_hosted_distro/together.md)
:::

:::{tab-item} fireworks
- [Start Fireworks Distribution](./self_hosted_distro/fireworks.md)
:::

::::

### Troubleshooting

- If you encounter any issues, search through our [GitHub Issues](https://github.com/meta-llama/llama-stack/issues), or file an new issue.
- Use `--port <PORT>` flag to use a different port number. For docker run, update the `-p <PORT>:<PORT>` flag.
