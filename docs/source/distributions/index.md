# Starting a Llama Stack
```{toctree}
:maxdepth: 3
:hidden:

importing_as_library
building_distro
configuration
```

<!-- self_hosted_distro/index -->
<!-- remote_hosted_distro/index -->
<!-- ondevice_distro/index -->

You can instantiate a Llama Stack in one of the following ways:
- **As a Library**: this is the simplest, especially if you are using an external inference service. See [Using Llama Stack as a Library](importing_as_library)
- **Docker**: we provide a number of pre-built Docker containers so you can start a Llama Stack server instantly. You can also build your own custom Docker container.
- **Conda**: finally, you can build a custom Llama Stack server using `llama stack build` containing the exact combination of providers you wish. We have provided various templates to make getting started easier.

Which templates / distributions to choose depends on the hardware you have for running LLM inference.

- **Do you have access to a machine with powerful GPUs?**
If so, we suggest:
  - {dockerhub}`distribution-remote-vllm` ([Guide](self_hosted_distro/remote-vllm))
  - {dockerhub}`distribution-meta-reference-gpu` ([Guide](self_hosted_distro/meta-reference-gpu))
  - {dockerhub}`distribution-tgi` ([Guide](self_hosted_distro/tgi))
  - {dockerhub} `distribution-nvidia` ([Guide](self_hosted_distro/nvidia))

- **Are you running on a "regular" desktop machine?**
If so, we suggest:
  - {dockerhub}`distribution-ollama` ([Guide](self_hosted_distro/ollama))

- **Do you have an API key for a remote inference provider like Fireworks, Together, etc.?** If so, we suggest:
  - {dockerhub}`distribution-together` ([Guide](remote_hosted_distro/index))
  - {dockerhub}`distribution-fireworks` ([Guide](remote_hosted_distro/index))

- **Do you want to run Llama Stack inference on your iOS / Android device** If so, we suggest:
  - [iOS SDK](ondevice_distro/ios_sdk)
  - [Android](ondevice_distro/android_sdk)

You can also build your own [custom distribution](building_distro).
