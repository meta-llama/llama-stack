# API Providers

A Provider is what makes the API real -- they provide the actual implementation backing the API.

As an example, for Inference, we could have the implementation be backed by open source libraries like `[ torch | vLLM | TensorRT ]` as possible options.

A provider can also be just a pointer to a remote REST service -- for example, cloud providers or dedicated inference providers could serve these APIs.

```{toctree}
:maxdepth: 1

new_api_provider
memory_api
```
