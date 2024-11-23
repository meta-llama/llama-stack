# Starting a Llama Stack
```{toctree}
:maxdepth: 3
:hidden:

importing_as_library
self_hosted_distro/index
remote_hosted_distro/index
building_distro
ondevice_distro/index
```

You can start a Llama Stack server using "distributions" (see [Concepts](../concepts/index)) in one of the following ways:
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
  - [distribution-together](remote_hosted_distro/index)
  - [distribution-fireworks](remote_hosted_distro/index)

- **Do you want to run Llama Stack inference on your iOS / Android device** If so, we suggest:
  - [iOS](ondevice_distro/ios_sdk)
  - Android (coming soon)

You can also build your own [custom distribution](building_distro).
