#!/usr/bin/env bash

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
OLLAMA_MODEL_ALIAS="llama3.2:3b-instruct-fp16"
OLLAMA_URL="http://localhost:11434"
CONTAINER_ENGINE=""

# Functions

print_banner() {
  echo -e "${BOLD}==================================================${NC}"
  echo -e "${BOLD}    Llama Stack Ollama Distribution Setup    ${NC}"
  echo -e "${BOLD}==================================================${NC}"
}

check_command() {
  command -v "$1" &> /dev/null
}

# Function to check prerequisites
check_prerequisites() {
  echo -e "\n${BOLD}Checking prerequisites...${NC}"

  # Check for container engine (Docker or Podman)
  if check_command docker; then
    echo -e "${GREEN}✓${NC} Docker is installed"
    CONTAINER_ENGINE="docker"
  elif check_command podman; then
    echo -e "${GREEN}✓${NC} Podman is installed"
    CONTAINER_ENGINE="podman"
  else
    echo -e "${RED}Error: Neither Docker nor Podman is installed. Please install one of them first.${NC}"
    echo "Visit https://docs.docker.com/get-docker/ or https://podman.io/getting-started/installation for installation instructions."
    exit 1
  fi

  # Check Python and pip
  if check_command python3; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [[ $(echo "$PYTHON_VERSION >= 3.10" | bc -l) -eq 1 ]]; then
      echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION is installed"
      HAS_PYTHON=true
    else
      echo -e "${YELLOW}Warning: Python $PYTHON_VERSION detected. Python 3.10+ recommended.${NC}"
      HAS_PYTHON=false
    fi
  else
    echo -e "${YELLOW}Warning: Python 3 is not found. Will use container for operations.${NC}"
    HAS_PYTHON=false
  fi

  # Check pip
  if [ "$HAS_PYTHON" = true ]; then
    if check_command pip || check_command pip3; then
      echo -e "${GREEN}✓${NC} pip is installed"
      HAS_PIP=true
    else
      echo -e "${YELLOW}Warning: pip is not found. Will use container for operations.${NC}"
      HAS_PIP=false
      HAS_PYTHON=false
    fi
  fi
}

# Function to install Ollama
install_ollama() {
  echo -e "\n${BOLD}Installing Ollama...${NC}"

  if check_command ollama; then
    echo -e "${GREEN}✓${NC} Ollama is already installed"
  else
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh

    if [ $? -eq 0 ]; then
      echo -e "${GREEN}✓${NC} Ollama installed successfully"
    else
      echo -e "${RED}Error: Failed to install Ollama.${NC}"
      exit 1
    fi
  fi
}

# Function to start Ollama server
start_ollama() {
  echo -e "\n${BOLD}Starting Ollama server...${NC}"

  # Check if Ollama is already running
  if curl -s "$OLLAMA_URL" &> /dev/null; then
    echo -e "${GREEN}✓${NC} Ollama server is already running"
  else
    echo "Starting Ollama server..."
    ollama serve &

    # Wait for Ollama server to start
    MAX_RETRIES=30
    RETRY_COUNT=0

    while ! curl -s "$OLLAMA_URL" &> /dev/null; do
      sleep 1
      RETRY_COUNT=$((RETRY_COUNT + 1))

      if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo -e "${RED}Error: Ollama server failed to start after $MAX_RETRIES seconds.${NC}"
        exit 1
      fi
    done

    echo -e "${GREEN}✓${NC} Ollama server started successfully"
  fi
}

# Function to pull models
pull_models() {
  echo -e "\n${BOLD}Pulling and running Llama model in Ollama...${NC}"

  # Pull model
  echo "Pulling $INFERENCE_MODEL model as $OLLAMA_MODEL_ALIAS..."
  ollama pull $OLLAMA_MODEL_ALIAS
  if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to pull $OLLAMA_MODEL_ALIAS model.${NC}"
    exit 1
  fi

  # Kill any existing model processes
  pkill -f "ollama run $OLLAMA_MODEL_ALIAS" || true

  # Start model in background
  echo "Starting inference model..."
  nohup ollama run $OLLAMA_MODEL_ALIAS --keepalive 60m > /dev/null 2>&1 &

  # Verify model is running by checking the Ollama API
  echo "Waiting for model to start (this may take a minute)..."

  MAX_RETRIES=30
  RETRY_DELAY=2

  # Wait for model to appear in the Ollama API
  for i in $(seq 1 $MAX_RETRIES); do
    echo -n "."
    MODELS_RUNNING=$(curl -s "$OLLAMA_URL/api/ps" | grep -E "$OLLAMA_MODEL_ALIAS" | wc -l)

    if [ "$MODELS_RUNNING" -ge 1 ]; then
      echo -e "\n${GREEN}✓${NC} Model is running successfully"
      break
    fi

    if [ $i -eq $MAX_RETRIES ]; then
      echo -e "\n${RED}Error: Model failed to start within the expected time.${NC}"
      exit 1
    fi

    sleep $RETRY_DELAY
  done
}

# Function to set up Python environment and install llama-stack-client
setup_llama_stack_cli() {
  echo -e "\n${BOLD}Setting up llama-stack environment...${NC}"

  # Create virtual environment
  echo "Creating Python virtual environment..."
  VENV_DIR="$HOME/.venv/llama-stack"

  if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
  else
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
      echo -e "${RED}Error: Failed to create virtual environment.${NC}"
      exit 1
    else
      echo -e "${GREEN}✓${NC} Virtual environment created successfully"
    fi
  fi

  # Activate virtual environment and install packages
  source "$VENV_DIR/bin/activate"

  echo "Installing llama-stack-client..."
  pip install --upgrade pip
  pip install llama-stack-client

  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} llama-stack-client installed successfully"

    # Configure the client to point to the correct server
    echo "Configuring llama-stack-client..."
    llama-stack-client configure --endpoint "http://localhost:$PORT"

    if [ $? -eq 0 ]; then
      echo -e "${GREEN}✓${NC} llama-stack-client configured to use http://localhost:$PORT"
      # Set environment variable for CLI use
      export LLAMA_STACK_BASE_URL="http://localhost:$PORT"
      # Add to shell config if it exists
      if [ -f "$HOME/.bashrc" ]; then
        grep -q "LLAMA_STACK_BASE_URL" "$HOME/.bashrc" || echo "export LLAMA_STACK_BASE_URL=\"http://localhost:$PORT\"" >> "$HOME/.bashrc"
      elif [ -f "$HOME/.zshrc" ]; then
        grep -q "LLAMA_STACK_BASE_URL" "$HOME/.zshrc" || echo "export LLAMA_STACK_BASE_URL=\"http://localhost:$PORT\"" >> "$HOME/.zshrc"
      fi
    else
      echo -e "${YELLOW}Warning: Failed to configure llama-stack-client. You may need to run 'llama-stack-client configure --endpoint http://localhost:$PORT' manually.${NC}"
    fi
  else
    echo -e "${RED}Error: Failed to install llama-stack-client.${NC}"
    exit 1
  fi
}

# Function to run a test inference
run_test_inference() {
  # Run a test inference to verify everything is working
  echo -e "\n${BOLD}Running test inference...${NC}"

  # Show the query being sent
  TEST_QUERY="hello, what model are you?"
  echo -e "${BOLD}Query:${NC} \"$TEST_QUERY\""

  # Send the query and capture the result
  echo -e "${BOLD}Sending request...${NC}"
  TEST_RESULT=$(llama-stack-client inference chat-completion --message "$TEST_QUERY" 2>&1)

  # Display the full result
  echo -e "\n${BOLD}Response:${NC}"
  echo "$TEST_RESULT"

  if [[ $? -eq 0 && "$TEST_RESULT" == *"content"* ]]; then
    echo -e "\n${GREEN}✓${NC} Test inference successful! Response received from the model."
    echo -e "${BOLD}Everything is working correctly!${NC}"
  else
    echo -e "\n${YELLOW}Warning: Test inference might have failed.${NC}"
    echo -e "You can try running a test manually after activation:"
    echo -e "${YELLOW}source $VENV_DIR/bin/activate${NC}"
    echo -e "${YELLOW}llama-stack-client inference chat-completion --message \"hello, what model are you?\"${NC}"
  fi
}

# Function to run the llama-stack server
run_llama_stack() {
  echo -e "\n${BOLD}Starting Llama Stack server...${NC}"

  mkdir -p "$HOME/.llama"

  # Check if container already exists
  CONTAINER_NAME="llama-stack-ollama"
  CONTAINER_EXISTS=false
  CONTAINER_RUNNING=false

  if [ "$CONTAINER_ENGINE" = "docker" ]; then
    if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
      CONTAINER_EXISTS=true
      if docker ps --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
        CONTAINER_RUNNING=true
      fi
    fi
  elif [ "$CONTAINER_ENGINE" = "podman" ]; then
    if podman ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
      CONTAINER_EXISTS=true
      if podman ps --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
        CONTAINER_RUNNING=true
      fi
    fi
  fi

  # Handle existing container
  if [ "$CONTAINER_EXISTS" = true ]; then
    if [ "$CONTAINER_RUNNING" = true ]; then
      echo -e "${YELLOW}Container $CONTAINER_NAME is already running${NC}"
      echo -e "${GREEN}✓${NC} Llama Stack server is already running"

      echo -e "\n${BOLD}Access Information:${NC}"
      echo -e "  • API URL: ${GREEN}http://localhost:$PORT${NC}"
      echo -e "  • Inference Model: ${GREEN}$INFERENCE_MODEL${NC}"
      echo -e "  • Ollama URL: ${GREEN}$OLLAMA_URL${NC}"

      echo -e "\n${BOLD}Management Commands:${NC}"
      echo -e "  • Stop Llama Stack:  ${YELLOW}${CONTAINER_ENGINE} stop $CONTAINER_NAME${NC}"
      echo -e "  • Start Llama Stack: ${YELLOW}${CONTAINER_ENGINE} start $CONTAINER_NAME${NC}"
      echo -e "  • View Logs:         ${YELLOW}${CONTAINER_ENGINE} logs $CONTAINER_NAME${NC}"
      echo -e "  • Stop Ollama:       ${YELLOW}pkill ollama${NC}"

      # Run a test inference
      run_test_inference

      return 0
    else
      echo -e "${YELLOW}Container $CONTAINER_NAME exists but is not running${NC}"
      if [ "$CONTAINER_ENGINE" = "docker" ]; then
        echo "Removing existing container..."
        docker rm $CONTAINER_NAME
      elif [ "$CONTAINER_ENGINE" = "podman" ]; then
        echo "Removing existing container..."
        podman rm $CONTAINER_NAME
      fi
    fi
  fi

  # Set the correct host value based on container engine
  if [ "$CONTAINER_ENGINE" = "docker" ]; then
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
      # Linux with Docker should use host network
      echo "Running Llama Stack server on Linux with Docker..."
      docker run -d \
        --name $CONTAINER_NAME \
        -p $PORT:$PORT \
        -v "$HOME/.llama:/root/.llama" \
        --network=host \
        llamastack/distribution-ollama \
        --port $PORT \
        --env INFERENCE_MODEL=$INFERENCE_MODEL \
        --env OLLAMA_URL=http://localhost:11434
    else
      # macOS/Windows with Docker should use host.docker.internal
      echo "Running Llama Stack server with Docker..."
      docker run -d \
        --name $CONTAINER_NAME \
        -p $PORT:$PORT \
        -v "$HOME/.llama:/root/.llama" \
        llamastack/distribution-ollama \
        --port $PORT \
        --env INFERENCE_MODEL=$INFERENCE_MODEL \
        --env OLLAMA_URL=http://host.docker.internal:11434
    fi
  elif [ "$CONTAINER_ENGINE" = "podman" ]; then
    # Check podman version for proper host naming
    PODMAN_VERSION=$(podman --version | awk '{print $3}')
    if [[ $(echo "$PODMAN_VERSION >= 4.7.0" | bc -l) -eq 1 ]]; then
      HOST_NAME="host.docker.internal"
    else
      HOST_NAME="host.containers.internal"
    fi

    echo "Running Llama Stack server with Podman..."
    podman run -d \
      --name $CONTAINER_NAME \
      -p $PORT:$PORT \
      -v "$HOME/.llama:/root/.llama:Z" \
      llamastack/distribution-ollama \
      --port $PORT \
      --env INFERENCE_MODEL=$INFERENCE_MODEL \
      --env OLLAMA_URL=http://$HOST_NAME:11434
  fi

  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Llama Stack server started successfully"

    echo -e "\n${BOLD}Setup Complete!${NC}"
    echo -e "\n${BOLD}Access Information:${NC}"
    echo -e "  • API URL: ${GREEN}http://localhost:$PORT${NC}"
    echo -e "  • Inference Model: ${GREEN}$INFERENCE_MODEL${NC}"
    echo -e "  • Ollama URL: ${GREEN}$OLLAMA_URL${NC}"

    echo -e "\n${BOLD}Management Commands:${NC}"
    echo -e "  • Stop Llama Stack:  ${YELLOW}${CONTAINER_ENGINE} stop $CONTAINER_NAME${NC}"
    echo -e "  • Start Llama Stack: ${YELLOW}${CONTAINER_ENGINE} start $CONTAINER_NAME${NC}"
    echo -e "  • View Logs:         ${YELLOW}${CONTAINER_ENGINE} logs $CONTAINER_NAME${NC}"
    echo -e "  • Stop Ollama:       ${YELLOW}pkill ollama${NC}"

    echo -e "\n${BOLD}Using Llama Stack Client:${NC}"
    echo -e "1. Activate the virtual environment: ${YELLOW}source $VENV_DIR/bin/activate${NC}"
    echo -e "2. Set the server URL: ${YELLOW}export LLAMA_STACK_BASE_URL=http://localhost:$PORT${NC}"
    echo -e "3. Run client commands: ${YELLOW}llama-stack-client --help${NC}"

    # Run a test inference
    run_test_inference
  else
    echo -e "${RED}Error: Failed to start Llama Stack server.${NC}"
    exit 1
  fi
}

# Main installation flow
main() {
  print_banner
  check_prerequisites
  install_ollama
  start_ollama
  pull_models
  setup_llama_stack_cli
  run_llama_stack
}

# Run main function
main
