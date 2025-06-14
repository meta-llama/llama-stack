#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

[ -z "$BASH_VERSION" ] && {
  echo "This script must be run with bash" >&2
  exit 1
}

set -Eeuo pipefail

PORT=8321
OLLAMA_PORT=11434
MODEL_ALIAS="llama3.2:3b"
SERVER_IMAGE="llamastack/distribution-ollama:0.2.2"
WAIT_TIMEOUT=300

log(){ printf "\e[1;32m%s\e[0m\n" "$*"; }
die(){ printf "\e[1;31m❌ %s\e[0m\n" "$*" >&2; exit 1; }

wait_for_service() {
  local url="$1"
  local pattern="$2"
  local timeout="$3"
  local name="$4"
  local start ts
  log "⏳  Waiting for ${name}…"
  start=$(date +%s)
  while true; do
    if curl --retry 5 --retry-delay 1 --retry-max-time "$timeout" --retry-all-errors --silent --fail "$url" 2>/dev/null | grep -q "$pattern"; then
      break
    fi
    ts=$(date +%s)
    if (( ts - start >= timeout )); then
      return 1
    fi
    printf '.'
    sleep 1
  done
  return 0
}

usage() {
    cat << EOF
📚 Llama-Stack Deployment Script

Description:
    This script sets up and deploys Llama-Stack with Ollama integration in containers.
    It handles both Docker and Podman runtimes and includes automatic platform detection.

Usage:
    $(basename "$0") [OPTIONS]

Options:
    -p, --port PORT            Server port for Llama-Stack (default: ${PORT})
    -o, --ollama-port PORT     Ollama service port (default: ${OLLAMA_PORT})
    -m, --model MODEL          Model alias to use (default: ${MODEL_ALIAS})
    -i, --image IMAGE          Server image (default: ${SERVER_IMAGE})
    -t, --timeout SECONDS      Service wait timeout in seconds (default: ${WAIT_TIMEOUT})
    -h, --help                 Show this help message

For more information:
    Documentation: https://llama-stack.readthedocs.io/
    GitHub: https://github.com/meta-llama/llama-stack

Report issues:
    https://github.com/meta-llama/llama-stack/issues
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -o|--ollama-port)
            OLLAMA_PORT="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_ALIAS="$2"
            shift 2
            ;;
        -i|--image)
            SERVER_IMAGE="$2"
            shift 2
            ;;
        -t|--timeout)
            WAIT_TIMEOUT="$2"
            shift 2
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

if command -v docker &> /dev/null; then
  ENGINE="docker"
elif command -v podman &> /dev/null; then
  ENGINE="podman"
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
# If you need GPU passthrough under Podman on macOS, init the VM with libkrun:
#   CONTAINERS_MACHINE_PROVIDER=libkrun podman machine init
if [ "$ENGINE" = "podman" ] && [ "$(uname -s)" = "Darwin" ]; then
  if ! podman info &>/dev/null; then
    log "⌛️ Initializing Podman VM…"
    podman machine init &>/dev/null || true
    podman machine start &>/dev/null || true

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
# 0. Create a shared network
###############################################################################
if ! $ENGINE network inspect llama-net >/dev/null 2>&1; then
  log "🌐  Creating network…"
  $ENGINE network create llama-net >/dev/null 2>&1
fi

###############################################################################
# 1. Ollama
###############################################################################
log "🦙  Starting Ollama…"
$ENGINE run -d "${PLATFORM_OPTS[@]}" --name ollama-server \
  --network llama-net \
  -p "${OLLAMA_PORT}:${OLLAMA_PORT}" \
  ollama/ollama > /dev/null 2>&1

if ! wait_for_service "http://localhost:${OLLAMA_PORT}/" "Ollama" "$WAIT_TIMEOUT" "Ollama daemon"; then
  log "❌  Ollama daemon did not become ready in ${WAIT_TIMEOUT}s; dumping container logs:"
  $ENGINE logs --tail 200 ollama-server
  die "Ollama startup failed"
fi

log "📦  Ensuring model is pulled: ${MODEL_ALIAS}…"
if ! $ENGINE exec ollama-server ollama pull "${MODEL_ALIAS}" > /dev/null 2>&1; then
  log "❌  Failed to pull model ${MODEL_ALIAS}; dumping container logs:"
  $ENGINE logs --tail 200 ollama-server
  die "Model pull failed"
fi

###############################################################################
# 2. Llama‑Stack
###############################################################################
cmd=( run -d "${PLATFORM_OPTS[@]}" --name llama-stack \
      --network llama-net \
      -p "${PORT}:${PORT}" \
      "${SERVER_IMAGE}" --port "${PORT}" \
      --env INFERENCE_MODEL="${MODEL_ALIAS}" \
      --env OLLAMA_URL="http://ollama-server:${OLLAMA_PORT}" )

log "🦙  Starting Llama‑Stack…"
$ENGINE "${cmd[@]}" > /dev/null 2>&1

if ! wait_for_service "http://127.0.0.1:${PORT}/v1/health" "OK" "$WAIT_TIMEOUT" "Llama-Stack API"; then
  log "❌  Llama-Stack did not become ready in ${WAIT_TIMEOUT}s; dumping container logs:"
  $ENGINE logs --tail 200 llama-stack
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
