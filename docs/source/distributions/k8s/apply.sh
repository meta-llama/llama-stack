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

set -euo pipefail
set -x

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
