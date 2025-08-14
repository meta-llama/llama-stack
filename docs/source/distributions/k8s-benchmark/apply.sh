#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Deploys the benchmark-specific components on top of the base k8s deployment (../k8s/apply.sh).

export MOCK_INFERENCE_PORT=8080
export STREAM_DELAY_SECONDS=0.005

export POSTGRES_USER=llamastack
export POSTGRES_DB=llamastack
export POSTGRES_PASSWORD=llamastack

export INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct
export SAFETY_MODEL=meta-llama/Llama-Guard-3-1B

export MOCK_INFERENCE_MODEL=mock-inference

# Use llama-stack-benchmark-service as the benchmark server
export LOCUST_HOST=http://llama-stack-benchmark-service:8323
export LOCUST_BASE_PATH=/v1/openai/v1

# Use vllm-service as the benchmark server
# export LOCUST_HOST=http://vllm-server:8000
# export LOCUST_BASE_PATH=/v1


export BENCHMARK_INFERENCE_MODEL=$INFERENCE_MODEL

set -euo pipefail
set -x

# Deploy benchmark-specific components
# Deploy OpenAI mock server
kubectl create configmap openai-mock --from-file=openai-mock-server.py \
  --dry-run=client -o yaml | kubectl apply --validate=false -f -

envsubst < openai-mock-deployment.yaml | kubectl apply --validate=false -f -

# Create configmap with our custom stack config
kubectl create configmap llama-stack-config --from-file=stack_run_config.yaml \
  --dry-run=client -o yaml > stack-configmap.yaml

kubectl apply --validate=false -f stack-configmap.yaml

# Deploy our custom llama stack server (overriding the base one)
envsubst < stack-k8s.yaml.template | kubectl apply --validate=false -f -

# Deploy Locust load testing
kubectl create configmap locust-script --from-file=locustfile.py \
  --dry-run=client -o yaml | kubectl apply --validate=false -f -

envsubst < locust-k8s.yaml | kubectl apply --validate=false -f -
