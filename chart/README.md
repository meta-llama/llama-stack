
# Llama Stack Helm Chart

This Helm chart is designed to install the Llama Stack, a comprehensive platform for llama-related tasks.

The chart provides a convenient way to deploy and manage the Llama Stack on Kubernetes or OpenShift clusters. It offers flexibility in customizing the deployment by allowing users to modify values such as image repositories, probe configurations, resource limits, and more.

Optionally, the chart also supports the installation of the llama-stack-playground, which provides a web-based interface for interacting with the Llama Stack.

## Quick Start

Create a `local-values.yaml` file with the following:

> **Note**
> Chart currently only supports `vllm` framework directly. But other distributions can be used by modifying the `env` directly.

```yaml
vllm:
  url: "https://<MY_VLLM_INSTANCE>:443/v1"
  inferenceModel: "meta-llama/Llama-3.1-8B-Instruct"
  apiKey: xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Login to Kubernetes through the CLI and run:

```sh
helm upgrade -i ollama-stack . -f local-values.yaml
```

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| autoscaling.enabled | bool | `false` |  |
| autoscaling.maxReplicas | int | `100` |  |
| autoscaling.minReplicas | int | `1` |  |
| autoscaling.targetCPUUtilizationPercentage | int | `80` |  |
| distribution | string | `"distribution-remote-vllm"` |  |
| image.pullPolicy | string | `"Always"` |  |
| image.repository | string | `"docker.io/llamastack/{{ $.Values.distribution }}"` |  |
| image.tag | string | `"0.1.6"` |  |
| ingress.annotations | object | `{}` |  |
| ingress.className | string | `""` |  |
| ingress.enabled | bool | `true` |  |
| ingress.hosts[0].host | string | `"chart-example.local"` |  |
| ingress.hosts[0].paths[0].path | string | `"/"` |  |
| ingress.hosts[0].paths[0].pathType | string | `"ImplementationSpecific"` |  |
| ingress.tls | list | `[]` |  |
| livenessProbe.httpGet.path | string | `"/v1/health"` |  |
| livenessProbe.httpGet.port | int | `5001` |  |
| podAnnotations | object | `{}` |  |
| podLabels | object | `{}` |  |
| podSecurityContext | object | `{}` |  |
| readinessProbe.httpGet.path | string | `"/v1/health"` |  |
| readinessProbe.httpGet.port | int | `5001` |  |
| replicaCount | int | `1` |  |
| resources.limits.cpu | string | `"100m"` |  |
| resources.limits.memory | string | `"500Mi"` |  |
| resources.requests.cpu | string | `"100m"` |  |
| resources.requests.memory | string | `"500Mi"` |  |
| route | object | `{"annotations":{},"enabled":false,"host":"","path":"","tls":{"enabled":true,"insecureEdgeTerminationPolicy":"Redirect","termination":"edge"}}` | Enable creation of the OpenShift Route object (This should be used instead of ingress on OpenShift) |
| route.annotations | object | `{}` | Additional custom annotations for the route |
| route.host | string | Set by OpenShift | The hostname for the route |
| route.path | string | `""` | The path for the OpenShift route |
| route.tls.enabled | bool | `true` | Enable secure route settings |
| route.tls.insecureEdgeTerminationPolicy | string | `"Redirect"` | Insecure route termination policy |
| route.tls.termination | string | `"edge"` | Secure route termination policy |
| runConfig.enabled | bool | `false` |  |
| service.port | int | `5001` |  |
| service.type | string | `"ClusterIP"` |  |
| serviceAccount.annotations | object | `{}` |  |
| serviceAccount.automount | bool | `true` |  |
| serviceAccount.create | bool | `false` |  |
| serviceAccount.name | string | `""` |  |
| startupProbe.failureThreshold | int | `30` |  |
| startupProbe.httpGet.path | string | `"/v1/health"` |  |
| startupProbe.httpGet.port | int | `5001` |  |
| startupProbe.initialDelaySeconds | int | `40` |  |
| startupProbe.periodSeconds | int | `10` |  |
| telemetry.enabled | bool | `false` |  |
| telemetry.serviceName | string | `"otel-collector.openshift-opentelemetry-operator.svc.cluster.local:4318"` |  |
| telemetry.sinks | string | `"console,sqlite,otel"` |  |
| vllm.inferenceModel | string | `"llama2-7b-chat"` |  |
| vllm.url | string | `"http://vllm-server"` |  |
| yamlConfig | string | `"/config/run.yaml"` |  |
