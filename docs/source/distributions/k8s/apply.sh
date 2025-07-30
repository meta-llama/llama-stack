#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Check if NGC_API_KEY is provided as argument
if [ -n "$1" ]; then
  export NGC_API_KEY=$1
  echo "Using NGC API key provided as argument."
fi

export POSTGRES_USER=llamastack
export POSTGRES_DB=llamastack
export POSTGRES_PASSWORD=llamastack

export INFERENCE_MODEL=meta-llama/Llama-3.1-8B-Instruct
export CODE_MODEL=bigcode/starcoder2-7b

# Set USE_EBS to false if you don't have permission to use EKS EBS
export USE_EBS=${USE_EBS:-false}

# HF_TOKEN should be set by the user; base64 encode it for the secret
if [ -n "${HF_TOKEN:-}" ]; then
  export HF_TOKEN_BASE64=$(echo -n "$HF_TOKEN" | base64)
else
  echo "ERROR: HF_TOKEN not set. You need it for vLLM to download models from Hugging Face."
  exit 1
fi

# NGC_API_KEY should be set by the user or provided as argument; base64 encode it for the secret
if [ -n "${NGC_API_KEY:-}" ]; then
  export NGC_API_KEY_BASE64=$(echo -n "$NGC_API_KEY" | base64)
  # Create Docker config JSON for NGC image pull
  NGC_DOCKER_CONFIG="{\"auths\":{\"nvcr.io\":{\"username\":\"\$oauthtoken\",\"password\":\"$NGC_API_KEY\"}}}"
  export NGC_DOCKER_CONFIG_JSON=$(echo -n "$NGC_DOCKER_CONFIG" | base64)
else
  echo "ERROR: NGC_API_KEY not set. You need it for NIM to download models from NVIDIA."
  echo "Usage: $0 [your-ngc-api-key]"
  echo "You can either provide your NGC API key as an argument or set it as an environment variable."
  exit 1
fi

if [ -z "${LLAMA_STACK_UI_URL:-}" ]; then
  echo "ERROR: LLAMA_STACK_UI_URL not set. Should be set to the external URL of the UI (excluding port). You need it for Github login to work. Refer to https://llama-stack.readthedocs.io/en/latest/deploying/index.html#kubernetes-deployment-guide"
  exit 1
fi



# Apply the HF token secret if HF_TOKEN is provided
if [ -n "${HF_TOKEN:-}" ]; then
  envsubst < ./set-secret.yaml.template | kubectl apply -f -
fi

set -euo pipefail
set -x
# Apply templates with appropriate storage configuration based on USE_EBS setting
if [ "$USE_EBS" = "true" ]; then
  echo "Using EBS storage for persistent volumes"
  envsubst < ./vllm-k8s.yaml.template | kubectl apply -f -
  envsubst < ./llama-nim.yaml.template | kubectl apply -f -
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
  envsubst < ./llama-nim.yaml.template | kubectl apply -f -
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
