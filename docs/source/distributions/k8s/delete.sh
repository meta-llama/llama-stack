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

set -euo pipefail
set -x

# Delete resources in reverse order of creation to handle dependencies properly

# Delete UI deployment
envsubst < ./ui-k8s.yaml.template | kubectl delete -f - --ignore-not-found=true

# Delete ingress
envsubst < ./ingress-k8s.yaml.template | kubectl delete -f - --ignore-not-found=true

# Delete stack deployment
envsubst < ./stack-k8s.yaml.template | kubectl delete -f - --ignore-not-found=true

# Delete configmap
kubectl delete configmap llama-stack-config --ignore-not-found=true

# Delete chroma deployment
envsubst < ./chroma-k8s.yaml.template | kubectl delete -f - --ignore-not-found=true

# Delete postgres deployment
envsubst < ./postgres-k8s.yaml.template | kubectl delete -f - --ignore-not-found=true

# Delete vllm-safety deployment
envsubst < ./vllm-safety-k8s.yaml.template | kubectl delete -f - --ignore-not-found=true

# Delete vllm deployment
envsubst < ./vllm-k8s.yaml.template | kubectl delete -f - --ignore-not-found=true

# Delete the HF token secret if it exists
if [ -n "${HF_TOKEN:-}" ]; then
  envsubst < ./hf-token-secret.yaml.template | kubectl delete -f - --ignore-not-found=true
fi

echo "All LlamaStack Kubernetes resources have been deleted."
