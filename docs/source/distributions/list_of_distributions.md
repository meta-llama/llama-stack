# Available List of Distributions

Here are a list of distributions you can use to start a Llama Stack server that are provided out of the box.

## Selection of a Distribution / Template

Which templates / distributions to choose depends on the hardware you have for running LLM inference.

- **Do you want a hosted Llama Stack endpoint?** If so, we suggest leveraging our partners who host Llama Stack endpoints. Namely, _fireworks.ai_ and _together.xyz_.
  - Read more about it here - [Remote-Hosted Endpoints](remote_hosted_distro/index).


- **Do you have access to machines with GPUs?** If you wish to run Llama Stack locally or on a cloud instance and host your own Llama Stack endpoint, we suggest:
  - {dockerhub}`distribution-remote-vllm` ([Guide](self_hosted_distro/remote-vllm))
  - {dockerhub}`distribution-meta-reference-gpu` ([Guide](self_hosted_distro/meta-reference-gpu))
  - {dockerhub}`distribution-tgi` ([Guide](self_hosted_distro/tgi))
  - {dockerhub}`distribution-nvidia` ([Guide](self_hosted_distro/nvidia))

- **Are you running on a "regular" desktop or laptop ?** We suggest using the ollama template for quick prototyping and get started without having to worry about needing GPUs.
  - {dockerhub}`distribution-ollama` ([Guide](self_hosted_distro/ollama))

- **Do you have an API key for a remote inference provider like Fireworks, Together, etc.?**  If so, we suggest:
  - {dockerhub}`distribution-together` ([Guide](self_hosted_distro/together))
  - {dockerhub}`distribution-fireworks` ([Guide](self_hosted_distro/fireworks))

- **Do you want to run Llama Stack inference on your iOS / Android device?**  Lastly, we also provide templates for running Llama Stack inference on your iOS / Android device:
  - [iOS SDK](ondevice_distro/ios_sdk)
  - [Android](ondevice_distro/android_sdk)


- **If none of the above fit your needs, you can also build your own [custom distribution](building_distro.md).**

### Distribution Details

```{toctree}
:maxdepth: 1

remote_hosted_distro/index
self_hosted_distro/remote-vllm
self_hosted_distro/meta-reference-gpu
self_hosted_distro/tgi
self_hosted_distro/nvidia
self_hosted_distro/ollama
self_hosted_distro/together
self_hosted_distro/fireworks
```

### On-Device Distributions

```{toctree}
:maxdepth: 1

ondevice_distro/ios_sdk
ondevice_distro/android_sdk
```
