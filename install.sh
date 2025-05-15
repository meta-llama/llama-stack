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
usage() {
    cat << EOF
📚 Llama-Stack Deployment Script v${VERSION}

Description:
    This script sets up and deploys Llama-Stack with Ollama integration in containers.
    It handles both Docker and Podman runtimes and includes automatic platform detection.

Usage: 
    $(basename "$0") [OPTIONS]

Options:
    -p, --port PORT            Server port for Llama-Stack (default: ${DEFAULT_PORT})
    -o, --ollama-port PORT     Ollama service port (default: ${DEFAULT_OLLAMA_PORT})
    -m, --model MODEL          Model alias to use (default: ${DEFAULT_MODEL_ALIAS})
    -i, --image IMAGE          Server image (default: ${DEFAULT_SERVER_IMAGE})
    -t, --timeout SECONDS      Service wait timeout in seconds (default: ${DEFAULT_WAIT_TIMEOUT})
    -c, --config FILE         Config file path (default: ${CONFIG_FILE})
    -v, --verbose             Enable verbose output
    -h, --help               Show this help message
    --version                Show version information

Configuration:
    The script can be configured using either command-line arguments or a config file.
    Config file location: ${CONFIG_FILE}
    Configuration precedence: Command-line > Config file > Default values

Environment Requirements:
    - Docker or Podman installed and running
    - Network connectivity for pulling images
    - Available ports for services
    - Sufficient system resources for running containers

Examples:
    1. Basic usage with default settings:
       $ $(basename "$0")

    2. Custom ports and model:
       $ $(basename "$0") --port 8080 --ollama-port 11435 --model "llama3.2:7b"

    3. Using verbose mode with custom timeout:
       $ $(basename "$0") -v --timeout 600

    4. Specify custom server image:
       $ $(basename "$0") --image "llamastack/distribution-ollama:latest"

Configuration File Example:
    # Contents for ${CONFIG_FILE}
    PORT=8080
    OLLAMA_PORT=11435
    MODEL_ALIAS="llama3.2:7b"
    WAIT_TIMEOUT=600
    SERVER_IMAGE="llamastack/distribution-ollama:latest"

Services:
    1. Ollama Server
       - Runs the Ollama service for model hosting
       - Default port: ${DEFAULT_OLLAMA_PORT}
       - Container name: ollama-server

    2. Llama-Stack
       - Runs the main Llama-Stack service
       - Default port: ${DEFAULT_PORT}
       - Container name: llama-stack

Network:
    - Creates a Docker/Podman network named 'llama-net'
    - All containers are connected to this network
    - Internal communication uses container names as hostnames

Logs and Debugging:
    - Use -v flag for verbose output
    - Container logs are available using:
      $ docker/podman logs ollama-server
      $ docker/podman logs llama-stack

For more information:
    Documentation: https://llama-stack.readthedocs.io/
    GitHub: https://github.com/llamastack/llamastack

Report issues:
    https://github.com/llamastack/llamastack/issues
EOF
}