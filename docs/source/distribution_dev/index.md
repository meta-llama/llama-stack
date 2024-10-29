# Developer Guide

```{toctree}
:hidden:
:maxdepth: 1

building_distro
```

## Key Concepts

### API Provider
A Provider is what makes the API real -- they provide the actual implementation backing the API.

As an example, for Inference, we could have the implementation be backed by open source libraries like `[ torch | vLLM | TensorRT ]` as possible options.

A provider can also be just a pointer to a remote REST service -- for example, cloud providers or dedicated inference providers could serve these APIs.

### Distribution
A Distribution is where APIs and Providers are assembled together to provide a consistent whole to the end application developer. You can mix-and-match providers -- some could be backed by local code and some could be remote. As a hobbyist, you can serve a small model locally, but can choose a cloud provider for a large model. Regardless, the higher level APIs your app needs to work with don't need to change at all. You can even imagine moving across the server / mobile-device boundary as well always using the same uniform set of APIs for developing Generative AI applications.
