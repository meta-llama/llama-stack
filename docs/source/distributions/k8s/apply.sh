#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

export POSTGRES_USER=${POSTGRES_USER:-llamastack}
export POSTGRES_DB=${POSTGRES_DB:-llamastack}
export POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-llamastack}

export INFERENCE_MODEL=${INFERENCE_MODEL:-meta-llama/Llama-3.2-3B-Instruct}
export SAFETY_MODEL=${SAFETY_MODEL:-meta-llama/Llama-Guard-3-1B}

# HF_TOKEN should be set by the user; base64 encode it for the secret
if [ -n "${HF_TOKEN:-}" ]; then
  export HF_TOKEN_BASE64=$(echo -n "$HF_TOKEN" | base64)
else
  echo "ERROR: HF_TOKEN not set. You need it for vLLM to download models from Hugging Face."
  exit 1
fi

set -euo pipefail
set -x

# Apply the HF token secret if HF_TOKEN is provided
if [ -n "${HF_TOKEN:-}" ]; then
  envsubst < ./hf-token-secret.yaml.template | kubectl apply -f -
fi

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
