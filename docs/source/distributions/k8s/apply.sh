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
else
  echo "ERROR: HF_TOKEN not set. You need it for vLLM to download models from Hugging Face."
  exit 1
fi

if [ -z "${GITHUB_CLIENT_ID:-}" ]; then
  echo "ERROR: GITHUB_CLIENT_ID not set. You need it for Github login to work. Refer to https://llama-stack.readthedocs.io/en/latest/deploying/index.html#kubernetes-deployment-guide"
  exit 1
fi

if [ -z "${GITHUB_CLIENT_SECRET:-}" ]; then
  echo "ERROR: GITHUB_CLIENT_SECRET not set. You need it for Github login to work. Refer to https://llama-stack.readthedocs.io/en/latest/deploying/index.html#kubernetes-deployment-guide"
  exit 1
fi

if [ -z "${LLAMA_STACK_UI_URL:-}" ]; then
  echo "ERROR: LLAMA_STACK_UI_URL not set. Should be set to the external URL of the UI (excluding port). You need it for Github login to work. Refer to https://llama-stack.readthedocs.io/en/latest/deploying/index.html#kubernetes-deployment-guide"
  exit 1
fi




set -euo pipefail
set -x

# Apply the HF token secret if HF_TOKEN is provided
if [ -n "${HF_TOKEN:-}" ]; then
  envsubst < ./hf-token-secret.yaml.template | kubectl apply -f -
fi

# Apply templates with appropriate storage configuration based on USE_EBS setting
if [ "$USE_EBS" = "true" ]; then
  echo "Using EBS storage for persistent volumes"
  envsubst < ./vllm-k8s.yaml.template | kubectl apply -f -
  envsubst < ./vllm-safety-k8s.yaml.template | kubectl apply -f -
  envsubst < ./postgres-k8s.yaml.template | kubectl apply -f -
  envsubst < ./chroma-k8s.yaml.template | kubectl apply -f -

  kubectl create configmap llama-stack-config --from-file=stack_run_config.yaml \
    --dry-run=client -o yaml > stack-configmap.yaml

  kubectl apply -f stack-configmap.yaml

  envsubst < ./stack-k8s.yaml.template | kubectl apply -f -
  envsubst < ./ingress-k8s.yaml.template | kubectl apply -f -
  envsubst < ./ui-k8s.yaml.template | kubectl apply -f -
else
  echo "Using emptyDir for storage (data will not persist across pod restarts)"
  # Process templates to replace EBS storage with emptyDir
  envsubst < ./vllm-k8s.yaml.template | sed 's/persistentVolumeClaim:/emptyDir: {}/g' | sed '/claimName:/d' | kubectl apply -f -
  envsubst < ./vllm-safety-k8s.yaml.template | sed 's/persistentVolumeClaim:/emptyDir: {}/g' | sed '/claimName:/d' | kubectl apply -f -
  envsubst < ./postgres-k8s.yaml.template | sed 's/persistentVolumeClaim:/emptyDir: {}/g' | sed '/claimName:/d' | kubectl apply -f -
  envsubst < ./chroma-k8s.yaml.template | sed 's/persistentVolumeClaim:/emptyDir: {}/g' | sed '/claimName:/d' | kubectl apply -f -

  kubectl create configmap llama-stack-config --from-file=stack_run_config.yaml \
    --dry-run=client -o yaml > stack-configmap.yaml

  kubectl apply -f stack-configmap.yaml

  # Apply the same emptyDir transformation to the remaining templates
  envsubst < ./stack-k8s.yaml.template | sed 's/persistentVolumeClaim:/emptyDir: {}/g' | sed '/claimName:/d' | kubectl apply -f -
  envsubst < ./ingress-k8s.yaml.template | sed 's/persistentVolumeClaim:/emptyDir: {}/g' | sed '/claimName:/d' | kubectl apply -f -
  envsubst < ./ui-k8s.yaml.template | sed 's/persistentVolumeClaim:/emptyDir: {}/g' | sed '/claimName:/d' | kubectl apply -f -
fi
