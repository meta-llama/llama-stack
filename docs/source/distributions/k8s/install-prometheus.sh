#!/bin/bash
# Script to install prometheus-community/kube-prometheus-stack using Helm

# Exit immediately if a command exits with a non-zero status
set -e

# Add the Prometheus community Helm repository if it doesn't exist
if ! helm repo list | grep -q "prometheus-community"; then
  echo "Adding prometheus-community Helm repository..."
  helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
fi

# Update Helm repositories
echo "Updating Helm repositories..."
helm repo update

# Create namespace for monitoring if it doesn't exist
if ! kubectl get namespace monitoring &> /dev/null; then
  echo "Creating monitoring namespace..."
  kubectl create namespace monitoring
fi

# Install kube-prometheus-stack
echo "Installing kube-prometheus-stack..."
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set grafana.enabled=true \
  --set prometheus.enabled=true \
  --set alertmanager.enabled=true \
  --set prometheus.service.type=ClusterIP \
  --set grafana.service.type=ClusterIP \
  --set alertmanager.service.type=ClusterIP \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false

echo "kube-prometheus-stack has been installed successfully!"
echo "To access Grafana UI, run: kubectl port-forward svc/kube-prometheus-stack-1754164871-grafana 31509:80 -n prometheus"
echo "Default Grafana credentials - Username: admin, Password: prom-operator"
