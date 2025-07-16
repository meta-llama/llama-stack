# Starting a Llama Stack Server

You can run a Llama Stack server in one of the following ways:

## As a Library:

This is the simplest way to get started. Using Llama Stack as a library means you do not need to start a server. This is especially useful when you are not running inference locally and relying on an external inference service (eg. fireworks, together, groq, etc.) See [Using Llama Stack as a Library](importing_as_library)


## Container:

Another simple way to start interacting with Llama Stack is to just spin up a container (via Docker or Podman) which is pre-built with all the providers you need. We provide a number of pre-built images so you can start a Llama Stack server instantly. You can also build your own custom container. Which distribution to choose depends on the hardware you have. See [Selection of a Distribution](selection) for more details.


## Conda:

If you have a custom or an advanced setup or you are developing on Llama Stack you can also build a custom Llama Stack server. Using `llama stack build` and `llama stack run` you can build/run a custom Llama Stack server containing the exact combination of providers you wish. We have also provided various templates to make getting started easier. See [Building a Custom Distribution](building_distro) for more details.


## Kubernetes:

If you have built a container image and want to deploy it in a Kubernetes cluster instead of starting the Llama Stack server locally. See [Kubernetes Deployment Guide](kubernetes_deployment) for more details.


```{toctree}
:maxdepth: 1
:hidden:

importing_as_library
configuration
```
