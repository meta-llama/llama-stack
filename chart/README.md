# Llama Stack on Kubernetes

Llama Stack can be installed into any Kubernetes Cluster using [Helm](https://helm.sh/docs/intro/install/)

### Prerequisites

* A Kubernetes Cluster with appropriate Node (CPU, GPU, etc)
* [Helm](https://helm.sh/docs/intro/install/)

### Install

```bash
helm install llama-stack oci://ghcr.io/meta-llama/llama-stack --set distribution=ollama
```

### Chat with Llama

```bash
kubectl port-forward svc/llama-stack 8080:80
llama-stack-client inference chat-completion --message "Hello, what model are you?"
```
