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

# Explicitly set the platform for the host architecture
HOST_ARCH="$(uname -m)"
if [ "$HOST_ARCH" = "arm64" ]; then
  if [ "$ENGINE" = "docker" ]; then
    PLATFORM_OPTS=( --platform linux/amd64 )
  else
    PLATFORM_OPTS=( --os linux --arch amd64 )
  fi
else
  PLATFORM_OPTS=()
fi

# macOS + Podman: ensure VM is running before we try to launch containers
if [ "$ENGINE" = "podman" ] && [ "$(uname -s)" = "Darwin" ]; then
  if ! podman info &>/dev/null; then
    log "⌛️ Initializing Podman VM…"
    podman machine init -q &>/dev/null || true
    podman machine start -q &>/dev/null || true

    log "⌛️  Waiting for Podman API…"
    until podman info &>/dev/null; do
      sleep 1
    done
    log "✅  Podman VM is up"
  fi
fi

# Clean up any leftovers from earlier runs
for name in ollama-server llama-stack; do
  ids=$($ENGINE ps -aq --filter "name=^${name}$")
  if [ -n "$ids" ]; then
    log "⚠️   Found existing container(s) for '${name}', removing…"
    $ENGINE rm -f "$ids" > /dev/null 2>&1
  fi
done

###############################################################################
# 1. Ollama
###############################################################################
log "🦙  Starting Ollama…"

$ENGINE run -d "${PLATFORM_OPTS[@]}" --name ollama-server \
  -p "${OLLAMA_PORT}:11434" \
  ollama/ollama > /dev/null 2>&1

log "⏳  Waiting for Ollama daemon…"
if ! timeout "$WAIT_TIMEOUT" bash -c \
    "until curl -fsS http://localhost:${OLLAMA_PORT}/ 2>/dev/null | grep -q 'Ollama'; do sleep 1; done"; then
  log "❌  Ollama daemon did not become ready in ${WAIT_TIMEOUT}s; dumping container logs:"
  $ENGINE logs ollama-server --tail=200
  die "Ollama startup failed"
fi

log "📦  Ensuring model is pulled: ${MODEL_ALIAS}…"
$ENGINE exec ollama-server ollama pull "${MODEL_ALIAS}" > /dev/null 2>&1

###############################################################################
# 2. Llama‑Stack
###############################################################################
if [ "$ENGINE" = "docker" ]; then
  NET_OPTS=( -p "${PORT}:${PORT}" --add-host="${HOST_DNS}:host-gateway" )
elif [ "$ENGINE" = "podman" ]; then
  NET_OPTS=( --network host )
fi

cmd=( run -d "${PLATFORM_OPTS[@]}" --name llama-stack "${NET_OPTS[@]}" \
      "${SERVER_IMAGE}" --port "${PORT}" \
      --env INFERENCE_MODEL="${MODEL_ALIAS}" \
      --env OLLAMA_URL="http://${HOST_DNS}:${OLLAMA_PORT}" )

log "🦙📦  Starting Llama‑Stack…"
$ENGINE "${cmd[@]}" > /dev/null 2>&1

log "⏳  Waiting for Llama-Stack API…"
if ! timeout "$WAIT_TIMEOUT" bash -c \
  "until $ENGINE exec llama-stack curl -fsS http://127.0.0.1:${PORT}/v1/health 2>/dev/null | grep -q 'OK'; do printf '.'; sleep 1; done"; then
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
