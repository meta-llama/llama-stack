# Llama Stack Kubernetes Deployment Guide

This guide explains how to deploy Llama Stack on Kubernetes using the files in this directory.

## Prerequisites

Before you begin, ensure you have:

- A Kubernetes cluster up and running
- `kubectl` installed and configured to access your cluster
- `envsubst` command available (part of the `gettext` package)
- Hugging Face API token (required for downloading models)
- NVIDIA NGC API key (required for NIM models)
For the cluster setup, please do:
1. Install Kubernetes nvidia operator, this will enable the GPU features:
```bash
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.12.3/nvidia-device-plugin.yml
```
2. Install prometheus and grafana for gpu monitoring following [this guide](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/kube-prometheus.html).

## Environment Setup

The deployment requires several environment variables to be set:

```bash
# Required environment variables
export HF_TOKEN=your_hugging_face_token  # Required for vLLM to download models
export NGC_API_KEY=your_ngc_api_key      # Required for NIM to download models

# Optional environment variables with defaults
export INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct  # Default inference model
export CODE_MODEL=bigcode/starcoder2-7b                  # Default code model
export OLLAMA_MODEL=llama-guard3:1b                      # Default safety model
export USE_EBS=false                                     # Use EBS storage (true/false)
export TAVILY_SEARCH_API_KEY=your_tavily_api_key         # Optional for search functionality
```

## Deployment Steps

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/meta-llama/llama-stack.git
   git checkout k8s_demo
   cd llama-stack/docs/source/distributions/k8s
   ```

2. **Deploy the stack**:
   ```bash
   export NGC_API_KEY=your_ngc_api_key
   export HF_TOKEN=your_hugging_face_token
   ./apply.sh
   ```

The deployment process:
1. Creates Kubernetes secrets for authentication
2. Deploys all components:
   - vLLM server (inference)
   - Ollama safety service
   - Llama NIM (code model)
   - PostgreSQL database
   - Chroma vector database
   - Jaeger (distributed tracing)
   - Llama Stack server
   - UI service
   - Ingress configuration

## Storage Options

The deployment supports two storage options:

1. **EBS Storage** (persistent):
   - Set `USE_EBS=true` for persistent storage
   - Data will persist across pod restarts
   - Requires EBS CSI driver in your cluster

2. **emptyDir Storage** (non-persistent):
   - Default option (`USE_EBS=false`)
   - Data will be lost when pods restart
   - Useful for testing or when EBS is not available

## Accessing the Services

After deployment, you can access the services:

1. **Check available service endpoint**:
   ```bash
   kubectl get svc
   kubectl get svc -n prometheus
   ```

2. **Port forward to access locally**:
   - To access the UI at http://localhost:8322, do:
   ```bash
   kubectl port-forward svc/llama-stack-service 8321:8321
   ```
   - To use the llama-stack endpoint at http://localhost:8321, do:
   ```bash
   kubectl port-forward svc/llama-stack-service 8321:8321 -n prometheus
   ```
   - To check the grafana endpoint at http://localhost:31509, do:
   ```bash
   kubectl port-forward svc/kube-prometheus-stack-1754164871-grafana 31509:80 -n prometheus
   ```


## Configuration

### Model Configuration

You can customize the models used by change environment variables in `apply.sh`:

```bash
export INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct  # Change to your preferred model
export CODE_MODEL=bigcode/starcoder2-7b                  # Change to your preferred code model
export OLLAMA_MODEL=llama-guard3:1b                      # Change to your preferred safety model
```

### Stack Configuration

The stack configuration is defined in `stack_run_config.yaml`. This file configures:
- API providers
- Models
- Database connections
- Tool integrations

If you need to modify this configuration, edit the file before running `apply.sh`.

## Monitoring and Telemetry

### Prometheus Monitoring

The deployment includes Prometheus monitoring capabilities:

```bash
# Install Prometheus monitoring
./install-prometheus.sh
```

### Jaeger Tracing

The deployment includes Jaeger for distributed tracing:

1. **Access the Jaeger UI**:
   ```bash
   kubectl port-forward svc/jaeger 16686:16686
   ```
   Then open http://localhost:16686 in your browser.

2. **Trace Configuration**:
   - Traces are automatically sent from llama-stack to Jaeger
   - The service name is set to "llama-stack" by default
   - Traces include spans for API calls, model inference, and other operations

3. **Troubleshooting Traces**:
   - If traces are not appearing in Jaeger:
     - Verify Jaeger is running: `kubectl get pods | grep jaeger`
     - Check llama-stack logs: `kubectl logs -f deployment/llama-stack-server`
     - Ensure the OTLP endpoint is correctly configured in the stack configuration
     - Verify network connectivity between llama-stack and Jaeger

## Cleanup

To remove all deployed resources:

```bash
./delete.sh
```

This will:
1. Delete all deployments, services, and configmaps
2. Remove persistent volume claims
3. Delete secrets

## Troubleshooting

### Common Issues

1. **Secret creation fails**:
   - Ensure your HF_TOKEN and NGC_API_KEY are correctly set
   - Check for any existing secrets that might conflict

2. **Pods stuck in pending state**:
   - Check if your cluster has enough resources
   - For GPU-based deployments, ensure GPU nodes are available

3. **Models fail to download**:
   - Verify your HF_TOKEN and NGC_API_KEY are valid
   - Check pod logs for specific error messages:
     ```bash
     kubectl logs -f deployment/vllm-server
     kubectl logs -f deployment/llm-nim-code
     ```

4. **Services not accessible**:
   - Verify all pods are running:
     ```bash
     kubectl get pods
     ```
   - Check service endpoints:
     ```bash
     kubectl get endpoints
     ```

5. **Traces not appearing in Jaeger**:
   - Check if the Jaeger pod is running: `kubectl get pods | grep jaeger`
   - Verify the llama-stack server is waiting for Jaeger to be ready before starting
   - Check the telemetry configuration in `stack_run_config.yaml`
   - Ensure the OTLP endpoint is correctly set to `http://jaeger.default.svc.cluster.local:4318`

### Viewing Logs

```bash
# View logs for specific components
kubectl logs -f deployment/llama-stack-server
kubectl logs -f deployment/vllm-server
kubectl logs -f deployment/llama-stack-ui
kubectl logs -f deployment/jaeger
```

## Advanced Configuration

### Custom Resource Limits

You can modify the resource limits in the YAML template files before deployment:

- `vllm-k8s.yaml.template`: vLLM server resources
- `stack-k8s.yaml.template`: Llama Stack server resources
- `llama-nim.yaml.template`: NIM server resources
- `jaeger-k8s.yaml.template`: Jaeger server resources

## Additional Resources

- [Llama Stack Documentation](https://github.com/meta-llama/llama-stack)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Jaeger Tracing Documentation](https://www.jaegertracing.io/docs/)
