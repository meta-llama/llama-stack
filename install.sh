#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -Eeuo pipefail

PORT=8321
OLLAMA_PORT=11434
MODEL_ALIAS="llama3.2:3b"
SERVER_IMAGE="llamastack/distribution-ollama:0.2.2"
WAIT_TIMEOUT=300
NETWORK="llama-stack-net"

log(){ printf "\e[1;32m%s\e[0m\n" "$*"; }
die(){ printf "\e[1;31m❌ %s\e[0m\n" "$*" >&2; exit 1; }

if   command -v docker &> /dev/null; then ENGINE="docker"
elif command -v podman &> /dev/null; then ENGINE="podman"
else die "Docker or Podman is required. Install Docker: https://docs.docker.com/get-docker/ or Podman: https://podman.io/getting-started/installation"
fi

if ! $ENGINE network ls --filter name=^${NETWORK}$ --format '{{.Name}}' | grep -q "^${NETWORK}$"; then
  log "Creating Docker network: ${NETWORK}"
  $ENGINE network create "${NETWORK}"
fi

# Clean up any leftovers from earlier runs
for name in ollama-server llama-stack; do
  $ENGINE ps -aq --filter "name=^${name}$" | xargs -r $ENGINE rm -f
done

###############################################################################
# 1. Ollama
###############################################################################
log "🦙  Starting Ollama…"
$ENGINE run -d --name ollama-server \
  --network "${NETWORK}" \
  -p "${OLLAMA_PORT}:11434" \
  -v ollama-models:/root/.ollama \
  ollama/ollama >/dev/null

log "⏳  Waiting for Ollama daemon…"
timeout "$WAIT_TIMEOUT" bash -c \
  "until curl -fsS http://localhost:${OLLAMA_PORT}/ 2>/dev/null | grep -q 'Ollama'; do sleep 1; done" \
  || die "Ollama did not become ready in ${WAIT_TIMEOUT}s"

if ! $ENGINE exec ollama-server ollama list | grep -q "$MODEL_ALIAS"; then
  log "📦  Pulling model $MODEL_ALIAS…"
  $ENGINE exec ollama-server ollama pull "$MODEL_ALIAS"
fi

log "🚀  Launching model runtime…"
$ENGINE exec -d ollama-server ollama run "$MODEL_ALIAS" --keepalive 60m

###############################################################################
# 2. Llama‑Stack
###############################################################################
log "🦙📦  Starting Llama‑Stack…"
$ENGINE run -d --name llama-stack \
  --network "${NETWORK}" \
  -p "${PORT}:${PORT}" \
  "${SERVER_IMAGE}" \
  --port "${PORT}" \
  --env INFERENCE_MODEL="${MODEL_ALIAS}" \
  --env OLLAMA_URL="http://ollama-server:${OLLAMA_PORT}" >/dev/null

log "⏳  Waiting for Llama‑Stack API…"
timeout "$WAIT_TIMEOUT" bash -c \
  "until curl -fsS http://localhost:${PORT}/v1/health 2>/dev/null | grep -q 'OK'; do sleep 1; done" \
  || die "Llama‑Stack did not become ready in ${WAIT_TIMEOUT}s"

###############################################################################
# Done
###############################################################################
log ""
log "🎉  Llama‑Stack is ready!"
log "👉  API endpoint: http://localhost:${PORT}"

# Note: if you’re calling from another container on the “${NETWORK}” network,
#       you can use the internal DNS name http://llama-stack:${PORT}
 log ""
