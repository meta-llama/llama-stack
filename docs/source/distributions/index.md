# Starting a Llama Stack

As mentioned in the [Concepts](../concepts/index), Llama Stack Distributions are specific pre-packaged versions of the Llama Stack. These templates make it easy to get started quickly.

A Llama Stack Distribution can be consumed in two ways:
- **Docker**: we provide a number of pre-built Docker containers allowing you to get started instantly. If you are focused on application development, we recommend this option. You can also build your own custom Docker container.
- **Conda**: the `llama` CLI provides a simple set of commands to build, configure and run a Llama Stack server containing the exact combination of providers you wish. We have provided various templates to make getting started easier.

Which distribution to choose depends on the hardware you have for running LLM inference.

- **Do you have access to a machine with powerful GPUs?**
If so, we suggest:
  - [distribution-remote-vllm](self_hosted_distro/remote-vllm)
  - [distribution-meta-reference-gpu](self_hosted_distro/meta-reference-gpu)
  - [distribution-tgi](self_hosted_distro/tgi)

- **Are you running on a "regular" desktop machine?**
If so, we suggest:
  - [distribution-ollama](self_hosted_distro/ollama)

- **Do you have an API key for a remote inference provider like Fireworks, Together, etc.?** If so, we suggest:
  - [distribution-together](#remote-hosted-distributions)
  - [distribution-fireworks](#remote-hosted-distributions)

- **Do you want to run Llama Stack inference on your iOS / Android device** If so, we suggest:
  - [iOS](ondevice_distro/ios_sdk)
  - [Android](ondevice_distro/android_sdk) (coming soon)


## Remote-Hosted Distributions

Remote-Hosted distributions are available endpoints serving Llama Stack API that you can directly connect to.

| Distribution | Endpoint | Inference | Agents | Memory | Safety | Telemetry |
|-------------|----------|-----------|---------|---------|---------|------------|
| Together | [https://llama-stack.together.ai](https://llama-stack.together.ai) | remote::together | meta-reference | remote::weaviate | meta-reference | meta-reference |
| Fireworks | [https://llamastack-preview.fireworks.ai](https://llamastack-preview.fireworks.ai) | remote::fireworks | meta-reference | remote::weaviate | meta-reference | meta-reference |

You can use `llama-stack-client` to interact with these endpoints. For example, to list the available models served by the Fireworks endpoint:

```bash
$ pip install llama-stack-client
$ llama-stack-client configure --endpoint https://llamastack-preview.fireworks.ai
$ llama-stack-client models list
```

## On-Device Distributions

On-device distributions are Llama Stack distributions that run locally on your iOS / Android device.


## Building Your Own Distribution

<TODO> talk about llama stack build --image-type conda, etc.

### Prerequisites

```bash
$ git clone git@github.com:meta-llama/llama-stack.git
```


### Troubleshooting

- If you encounter any issues, search through our [GitHub Issues](https://github.com/meta-llama/llama-stack/issues), or file an new issue.
- Use `--port <PORT>` flag to use a different port number. For docker run, update the `-p <PORT>:<PORT>` flag.


```{toctree}
:maxdepth: 3

remote_hosted_distro/index
ondevice_distro/index
```
