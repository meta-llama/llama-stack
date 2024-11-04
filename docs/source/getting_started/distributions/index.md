# Llama Stack Distribution

A Distribution is where APIs and Providers are assembled together to provide a consistent whole to the end application developer. You can mix-and-match providers -- some could be backed by local code and some could be remote. As a hobbyist, you can serve a small model locally, but can choose a cloud provider for a large model. Regardless, the higher level APIs your app needs to work with don't need to change at all. You can even imagine moving across the server / mobile-device boundary as well always using the same uniform set of APIs for developing Generative AI applications.

We offer three types of distributions:

1. [Deployable Distribution](./deployable_distro/index.md): If you want to run Llama Stack inference on your local machine. 
2. [Hosted Distribution](./hosted_distro/index.md): If you want to connect to a remote hosted inference provider.
3. [On-device Distribution](./ondevice_distro/index.md): If you want to run Llama Stack inference on your iOS / Android device.

```{toctree}
:maxdepth: 1
:hidden:

deployable_distro/index
hosted_distro/index
ondevice_distro/index
```
