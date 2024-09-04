#!/bin/bash

LLAMA_MODELS_DIR=${LLAMA_MODELS_DIR:-}
LLAMA_TOOLCHAIN_DIR=${LLAMA_TOOLCHAIN_DIR:-}
TEST_PYPI_VERSION=${TEST_PYPI_VERSION:-}

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <distribution_id> <build_name> <docker_base> <pip_dependencies>
  echo "Example: $0 distribution_id my-fastapi-app python:3.9-slim 'fastapi uvicorn'
  exit 1
fi

distribution_id=$1
build_name="$2"
image_name="llamastack-$build_name"
docker_base=$3
pip_dependencies=$4

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
REPO_DIR=$(dirname $(dirname "$SCRIPT_DIR"))

TEMP_DIR=$(mktemp -d)

add_to_docker() {
  local input
  output_file="$TEMP_DIR/Dockerfile"
  if [ -t 0 ]; then
    printf '%s\n' "$1" >>"$output_file"
  else
    # If stdin is not a terminal, read from it (heredoc)
    cat >>"$output_file"
  fi
}

add_to_docker <<EOF
FROM $docker_base
WORKDIR /app

RUN apt-get update && apt-get install -y \
       iputils-ping net-tools iproute2 dnsutils telnet \
       curl wget telnet \
       procps psmisc lsof \
       traceroute \
       && rm -rf /var/lib/apt/lists/*

EOF

toolchain_mount="/app/llama-toolchain-source"
models_mount="/app/llama-models-source"

if [ -n "$LLAMA_TOOLCHAIN_DIR" ]; then
  if [ ! -d "$LLAMA_TOOLCHAIN_DIR" ]; then
    echo "${RED}Warning: LLAMA_TOOLCHAIN_DIR is set but directory does not exist: $LLAMA_TOOLCHAIN_DIR${NC}" >&2
    exit 1
  fi
  add_to_docker "RUN pip install $toolchain_mount"
else
  add_to_docker "RUN pip install llama-toolchain"
fi

if [ -n "$LLAMA_MODELS_DIR" ]; then
  if [ ! -d "$LLAMA_MODELS_DIR" ]; then
    echo "${RED}Warning: LLAMA_MODELS_DIR is set but directory does not exist: $LLAMA_MODELS_DIR${NC}" >&2
    exit 1
  fi

  add_to_docker <<EOF
RUN pip uninstall -y llama-models
RUN pip install $models_mount

EOF
fi

if [ -n "$pip_dependencies" ]; then
  add_to_docker "RUN pip install $pip_dependencies"
fi

add_to_docker <<EOF

# This would be good in production but for debugging flexibility lets not add it right now
# We need a more solid production ready entrypoint.sh anyway
#
# ENTRYPOINT ["python", "-m", "llama_toolchain.core.server"]

EOF

printf "Dockerfile created successfully in $TEMP_DIR/Dockerfile"
cat $TEMP_DIR/Dockerfile
printf "\n"

mounts=""
if [ -n "$LLAMA_TOOLCHAIN_DIR" ]; then
  mounts="$mounts -v $(readlink -f $LLAMA_TOOLCHAIN_DIR):$toolchain_mount"
fi
if [ -n "$LLAMA_MODELS_DIR" ]; then
  mounts="$mounts -v $(readlink -f $LLAMA_MODELS_DIR):$models_mount"
fi
set -x
podman build -t $image_name -f "$TEMP_DIR/Dockerfile" "$REPO_DIR" $mounts
set +x

printf "${GREEN}Succesfully setup Podman image. Configuring build...${NC}"
echo "You can run it with: podman run -p 8000:8000 $image_name"

$CONDA_PREFIX/bin/python3 -m llama_toolchain.cli.llama stack configure $distribution_id --name "$build_name" --type container
