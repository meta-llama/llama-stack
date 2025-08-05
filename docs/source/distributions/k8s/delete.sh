#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

export POSTGRES_USER=llamastack
export POSTGRES_DB=llamastack
export POSTGRES_PASSWORD=llamastack

export INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct
export SAFETY_MODEL=meta-llama/Llama-Guard-3-1B

# Set USE_EBS to false if you don't have permission to use EKS EBS
export USE_EBS=${USE_EBS:-false}

# HF_TOKEN should be set by the user; base64 encode it for the secret
if [ -n "${HF_TOKEN:-}" ]; then
  export HF_TOKEN_BASE64=$(echo -n "$HF_TOKEN" | base64)
fi

# NGC_API_KEY should be set by the user; base64 encode it for the secret
if [ -n "${NGC_API_KEY:-}" ]; then
  export NGC_API_KEY_BASE64=$(echo -n "$NGC_API_KEY" | base64)
  # Create Docker config JSON for NGC image pull
  NGC_DOCKER_CONFIG="{\"auths\":{\"nvcr.io\":{\"username\":\"\$oauthtoken\",\"password\":\"$NGC_API_KEY\"}}}"
  export NGC_DOCKER_CONFIG_JSON=$(echo -n "$NGC_DOCKER_CONFIG" | base64)
fi

# Define namespace - default to current namespace if not specified
export NAMESPACE=${NAMESPACE:-$(kubectl config view --minify -o jsonpath='{..namespace}')}
if [ -z "$NAMESPACE" ]; then
  export NAMESPACE="default"
fi

set -euo pipefail
set -x

# Delete resources in reverse order of creation to handle dependencies properly

echo "Starting comprehensive deletion of all LlamaStack resources..."

# Delete UI deployment and service
echo "Deleting UI resources..."
envsubst < ./ui-k8s.yaml.template | kubectl delete -f - --ignore-not-found=true
# Check for UI service template and delete if exists
if [ -f "./ui-service-k8s.yaml.template" ]; then
  envsubst < ./ui-service-k8s.yaml.template | kubectl delete -f - --ignore-not-found=true
fi



# Delete ingress
echo "Deleting ingress resources..."
envsubst < ./ingress-k8s.yaml.template | kubectl delete -f - --ignore-not-found=true

# Delete stack deployment
echo "Deleting stack deployment..."
envsubst < ./stack-k8s.yaml.template | kubectl delete -f - --ignore-not-found=true

# Delete configmaps
echo "Deleting configmaps..."
kubectl delete configmap llama-stack-config --ignore-not-found=true
# Check for stack configmap and delete if exists
if [ -f "./stack-configmap.yaml" ]; then
  kubectl delete -f ./stack-configmap.yaml --ignore-not-found=true
fi

# Delete chroma deployment
echo "Deleting chroma deployment..."
envsubst < ./chroma-k8s.yaml.template | kubectl delete -f - --ignore-not-found=true

# Delete postgres deployment
echo "Deleting postgres deployment..."
envsubst < ./postgres-k8s.yaml.template | kubectl delete -f - --ignore-not-found=true

# Delete llama-nim deployment
echo "Deleting llama-nim deployment..."
envsubst < ./llama-nim.yaml.template | kubectl delete -f - --ignore-not-found=true


# Delete ollama-safety deployment
echo "Deleting ollama-safety deployment..."
envsubst < ./ollama-safety-k8s.yaml.template | kubectl delete -f - --ignore-not-found=true

# Delete vllm deployment
echo "Deleting vllm deployment..."
envsubst < ./vllm-k8s.yaml.template | kubectl delete -f - --ignore-not-found=true

# Delete jaeger deployment
echo "Deleting jaeger deployment..."
envsubst < ./jaeger-k8s.yaml.template | kubectl delete -f - --ignore-not-found=true

# Delete the HF token secret if it exists
if [ -n "${HF_TOKEN:-}" ]; then
  echo "Deleting HF token secret..."
  envsubst < ./set-secret.yaml.template | kubectl delete -f - --ignore-not-found=true
fi

# Delete any other template files that might exist
echo "Checking for additional template files..."
for template in ./*.yaml.template; do
  if [ -f "$template" ]; then
    # Skip templates we've already processed
    if [[ "$template" != "./ui-k8s.yaml.template" &&
          "$template" != "./ingress-k8s.yaml.template" &&
          "$template" != "./stack-k8s.yaml.template" &&
          "$template" != "./chroma-k8s.yaml.template" &&
          "$template" != "./postgres-k8s.yaml.template" &&
          "$template" != "./llama-nim.yaml.template" &&
          "$template" != "./vllm-safety-k8s.yaml.template" &&
          "$template" != "./ollama-safety-k8s.yaml.template" &&
          "$template" != "./vllm-k8s.yaml.template" &&
          "$template" != "./jaeger-k8s.yaml.template" &&
          "$template" != "./set-secret.yaml.template" &&
          "$template" != "./ui-service-k8s.yaml.template" ]]; then
      echo "Deleting resources from $template..."
      envsubst < "$template" | kubectl delete -f - --ignore-not-found=true
    fi
  fi
done

# Delete any PersistentVolumeClaims created by the stack
echo "Deleting PersistentVolumeClaims..."
kubectl delete pvc -l app=llama-stack --ignore-not-found=true
kubectl delete pvc -l app=chroma --ignore-not-found=true
kubectl delete pvc -l app=postgres --ignore-not-found=true
kubectl delete pvc -l app=vllm --ignore-not-found=true
kubectl delete pvc -l app.kubernetes.io/name=ollama-safety --ignore-not-found=true

# Delete any remaining services
echo "Deleting any remaining services..."
kubectl delete service -l app=llama-stack --ignore-not-found=true
kubectl delete service -l app=chroma --ignore-not-found=true
kubectl delete service -l app=postgres --ignore-not-found=true
kubectl delete service -l app=vllm --ignore-not-found=true
kubectl delete service -l app=llama-nim --ignore-not-found=true
kubectl delete service -l app.kubernetes.io/name=ollama-safety --ignore-not-found=true
kubectl delete service -l app=jaeger --ignore-not-found=true

# Delete any remaining secrets
echo "Deleting any remaining secrets..."
kubectl delete secret hf-secret --ignore-not-found=true
kubectl delete secret ngc-secret --ignore-not-found=true
kubectl delete secret -l app=llama-stack --ignore-not-found=true

# Verify no resources remain
echo "Verifying deletion..."
REMAINING_RESOURCES=$(kubectl get all -l app=llama-stack 2>/dev/null)
if [ -z "$REMAINING_RESOURCES" ]; then
  echo "All LlamaStack Kubernetes resources have been successfully deleted."
else
  echo "Some LlamaStack resources may still exist. Please check manually with:"
  echo "kubectl get all -l app=llama-stack"
fi
