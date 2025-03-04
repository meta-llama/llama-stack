#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -e

# Color codes for output formatting
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Default values
PORT=5001
INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
SAFETY_MODEL="meta-llama/Llama-Guard-3-1B"
# PROMPT_GUARD_MODEL="meta-llama/Prompt-Guard-86M"  # Commented out as it may be deprecated

# Banner
echo -e "${BOLD}==================================================${NC}"
echo -e "${BOLD}    Llama Stack Meta Reference Installation    ${NC}"
echo -e "${BOLD}==================================================${NC}"

# Function to check prerequisites
check_prerequisites() {
  echo -e "\n${BOLD}Checking prerequisites...${NC}"

  # Check Docker
  if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed. Please install Docker first.${NC}"
    echo "Visit https://docs.docker.com/get-docker/ for installation instructions."
    exit 1
  fi
  echo -e "${GREEN}✓${NC} Docker is installed"

  # Check Python
  if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Warning: Python 3 is not found. Will use Docker for all operations.${NC}"
    HAS_PYTHON=false
  else
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [[ $(echo "$PYTHON_VERSION >= 3.10" | bc) -eq 1 ]]; then
      echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION is installed"
      HAS_PYTHON=true
    else
      echo -e "${YELLOW}Warning: Python $PYTHON_VERSION detected. Python 3.10+ recommended.${NC}"
      HAS_PYTHON=false
    fi
  fi

  # Check NVIDIA GPU
  if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Warning: NVIDIA GPU drivers not detected.${NC}"
    echo -e "${YELLOW}This distribution is designed to run on NVIDIA GPUs and may not work on your system.${NC}"
    echo -e "It may still be useful for testing the installation process, but model loading will likely fail."
    echo -e "For production use, please install on a system with NVIDIA GPUs and proper drivers."

    read -p "Do you want to continue anyway? This may not work! (y/N): " CONTINUE
    if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
      echo "Installation aborted."
      exit 1
    fi
    echo -e "${YELLOW}Continuing without NVIDIA GPU. Expect issues.${NC}"
  else
    echo -e "${GREEN}✓${NC} NVIDIA GPU detected"
  fi
}

# Function to set up Python environment and install llama-stack
setup_llama_stack_cli() {
  echo -e "\n${BOLD}Setting up llama-stack CLI...${NC}"

  if [ "$HAS_PYTHON" = true ]; then
    # Create virtual environment
    echo "Creating Python virtual environment..."
    VENV_DIR="$HOME/.venv/llama-stack"
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    # Install pip and llama-stack
    echo "Installing llama-stack package..."
    pip install --upgrade pip
    pip install llama-stack

    echo -e "${GREEN}✓${NC} llama-stack CLI installed in virtual environment"
    LLAMA_CMD="$VENV_DIR/bin/llama"
  else
    echo -e "${YELLOW}Using Docker for llama-stack CLI operations${NC}"
    LLAMA_CMD="docker run --rm -v $HOME/.llama:/root/.llama llamastack/distribution-meta-reference-gpu llama"
  fi
}

# Function to download models
download_models() {
  echo -e "\n${BOLD}Downloading Llama models...${NC}"

  # Prompt for META_URL if not provided
  echo -e "Please enter your META_URL for model downloads."
  echo -e "${YELLOW}Note: You can get this URL from Meta's website when you're approved for model access.${NC}"
  read -p "META_URL: " META_URL

  if [ -z "$META_URL" ]; then
    echo -e "${RED}No META_URL provided. Cannot download models.${NC}"
    exit 1
  fi

  echo "Downloading $INFERENCE_MODEL..."
  $LLAMA_CMD model download --source meta --model-id "$INFERENCE_MODEL" --meta-url "$META_URL"

  echo "Downloading $SAFETY_MODEL..."
  $LLAMA_CMD model download --source meta --model-id "$SAFETY_MODEL" --meta-url "$META_URL"

  # Prompt Guard model may be deprecated
  # echo "Downloading $PROMPT_GUARD_MODEL..."
  # $LLAMA_CMD model download --source meta --model-id "$PROMPT_GUARD_MODEL" --meta-url "$META_URL"

  echo -e "${GREEN}✓${NC} Models downloaded successfully"
}

# Function to run the Docker container
run_docker_container() {
  echo -e "\n${BOLD}Setting up Docker container...${NC}"

  # Pull the latest image
  echo "Pulling llamastack/distribution-meta-reference-gpu image..."
  docker pull llamastack/distribution-meta-reference-gpu

  # Run the container
  echo "Starting container on port $PORT..."

  # Check if NVIDIA GPU is available
  if command -v nvidia-smi &> /dev/null; then
    # With GPU
    echo "Using NVIDIA GPU for Docker container..."
    docker run \
      -d \
      --name llama-stack-meta \
      -p $PORT:$PORT \
      -v $HOME/.llama:/root/.llama \
      --gpus all \
      llamastack/distribution-meta-reference-gpu \
      --port $PORT \
      --env INFERENCE_MODEL=$INFERENCE_MODEL \
      --env SAFETY_MODEL=$SAFETY_MODEL
  else
    # Without GPU (may not work)
    echo -e "${YELLOW}Warning: Running without GPU support. This will likely fail for model loading!${NC}"
    docker run \
      -d \
      --name llama-stack-meta \
      -p $PORT:$PORT \
      -v $HOME/.llama:/root/.llama \
      llamastack/distribution-meta-reference-gpu \
      --port $PORT \
      --env INFERENCE_MODEL=$INFERENCE_MODEL \
      --env SAFETY_MODEL=$SAFETY_MODEL
  fi

  # Check if container started successfully
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Llama Stack Meta Reference is now running!"
    echo -e "\n${BOLD}Access Information:${NC}"
    echo -e "  • API URL: ${GREEN}http://localhost:$PORT${NC}"
    echo -e "  • Inference Model: ${GREEN}$INFERENCE_MODEL${NC}"
    echo -e "  • Safety Model: ${GREEN}$SAFETY_MODEL${NC}"
    echo -e "\n${BOLD}Management Commands:${NC}"
    echo -e "  • Stop server:  ${YELLOW}docker stop llama-stack-meta${NC}"
    echo -e "  • Start server: ${YELLOW}docker start llama-stack-meta${NC}"
    echo -e "  • View logs:    ${YELLOW}docker logs llama-stack-meta${NC}"
  else
    echo -e "${RED}Failed to start the container. Please check Docker logs.${NC}"
    exit 1
  fi
}

# Main installation flow
main() {
  check_prerequisites
  setup_llama_stack_cli
  download_models
  run_docker_container
}

# Run main function
main
