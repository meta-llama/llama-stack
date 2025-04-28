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

log(){ printf "\e[1;32m%s\e[0m\n" "$*"; }
die(){ printf "\e[1;31m❌ %s\e[0m\n" "$*" >&2; exit 1; }

if command -v docker &> /dev/null; then
  ENGINE="docker"
  HOST_DNS="host.docker.internal"
elif command -v podman &> /dev/null; then
  ENGINE="podman"
  HOST_DNS="host.containers.internal"
else
  die "Docker or Podman is required. Install Docker: https://docs.docker.com/get-docker/ or Podman: https://podman.io/getting-started/installation"
fi

# Clean up any leftovers from earlier runs
for name in ollama-server llama-stack; do
  ids=$($ENGINE ps -aq --filter "name=^${name}$")
  if [ -n "$ids" ]; then
    log "⚠️   Found existing container(s) for '${name}', removing..."
    $ENGINE rm -f "$ids"
  fi
done

###############################################################################
# 1. Ollama
###############################################################################
log "🦙  Starting Ollama…"
$ENGINE run -d --name ollama-server \
  -p "${OLLAMA_PORT}:11434" \
  ollama/ollama > /dev/null 2>&1

log "⏳  Waiting for Ollama daemon…"
if ! timeout "$WAIT_TIMEOUT" bash -c \
    "until curl -fsS http://localhost:${OLLAMA_PORT}/ 2>/dev/null | grep -q 'Ollama'; do sleep 1; done"; then
  log "❌  Ollama daemon did not become ready in ${WAIT_TIMEOUT}s; dumping container logs:"
  $ENGINE logs ollama-server --tail=200
  die "Ollama startup failed"
fi

log "📦  Ensuring model is pulled: ${MODEL_ALIAS}..."
$ENGINE exec ollama-server ollama pull "${MODEL_ALIAS}" > /dev/null 2>&1

###############################################################################
# 2. Llama‑Stack
###############################################################################
log "🦙📦  Starting Llama‑Stack…"
$ENGINE run -d --name llama-stack \
  -p "${PORT}:${PORT}" \
  --add-host="${HOST_DNS}:host-gateway" \
  "${SERVER_IMAGE}" \
  --port "${PORT}" \
  --env INFERENCE_MODEL="${MODEL_ALIAS}" \
  --env OLLAMA_URL="http://${HOST_DNS}:${OLLAMA_PORT}" > /dev/null 2>&1

log "⏳  Waiting for Llama-Stack API…"
if ! timeout "$WAIT_TIMEOUT" bash -c \
  "until curl -fsS http://localhost:${PORT}/v1/health 2>/dev/null | grep -q 'OK'; do sleep 1; done"; then
  log "❌  Llama-Stack did not become ready in ${WAIT_TIMEOUT}s; dumping container logs:"
  $ENGINE logs llama-stack --tail=200
  die "Llama-Stack startup failed"
fi

###############################################################################
# Done
###############################################################################
log ""
log "🎉  Llama‑Stack is ready!"
log "👉  API endpoint: http://localhost:${PORT}"
log "📖 Documentation: https://llama-stack.readthedocs.io/en/latest/references/index.html"
log "💻 To access the llama‑stack CLI, exec into the container:"
log "   $ENGINE exec -ti llama-stack bash"
log ""
