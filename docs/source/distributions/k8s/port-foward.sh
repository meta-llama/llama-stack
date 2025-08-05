#!/bin/bash

# Function to port-forward to a pod with fallback to service
port_forward_with_fallback() {
  local namespace=$1
  local label_selector=$2
  local service_name=$3
  local local_port=$4
  local pod_port=$5

  echo "Attempting to port-forward to pod with label $label_selector in namespace $namespace..."

  # Try to get pod name using the label selector
  POD_NAME=$(kubectl get pods -n $namespace -l $label_selector -o jsonpath="{.items[0].metadata.name}" 2>/dev/null)

  if [ -n "$POD_NAME" ]; then
    echo "Found pod: $POD_NAME. Port-forwarding $local_port:$pod_port..."
    kubectl port-forward -n $namespace $POD_NAME $local_port:$pod_port &
  else
    echo "No pods found with label $label_selector in namespace $namespace."
    echo "Falling back to service port-forwarding for $service_name..."
    kubectl port-forward -n $namespace svc/$service_name $local_port:$pod_port &
  fi
}

# Port-forward to llama-stack-ui pod or service
port_forward_with_fallback "default" "app.kubernetes.io/name=llama-stack-ui" "llama-stack-ui-service" 8322 8322

# Port-forward to llama-stack server pod or service
port_forward_with_fallback "default" "app.kubernetes.io/name=llama-stack,app.kubernetes.io/component=server" "llama-stack-service" 8321 8321

# Port-forward to jaeger query pod or service in observability namespace
port_forward_with_fallback "observability" "app.kubernetes.io/component=query,app.kubernetes.io/instance=jaeger-dev" "jaeger-dev-query" 16686 16686

# Port-forward to grafana pod or service in prometheus namespace
kubectl port-forward svc/kube-prometheus-stack-1754270486-grafana 3000:80 -n prometheus

echo "Port-forwarding started for all components."
echo "Access the services at:"
echo "  - Llama Stack UI: http://localhost:8322"
echo "  - Llama Stack API: http://localhost:8321"
echo "  - Jaeger UI: http://localhost:16686"
echo "  - Grafana: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all port-forwarding processes."

# Wait for all background processes to complete
wait
