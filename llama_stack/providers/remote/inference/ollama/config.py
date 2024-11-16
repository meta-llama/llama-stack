# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.distribution.datatypes import RemoteProviderConfig
from llama_stack.providers.utils.docker.service_config import DockerComposeServiceConfig


DEFAULT_OLLAMA_PORT = 11434


class OllamaImplConfig(RemoteProviderConfig):
    port: int = DEFAULT_OLLAMA_PORT

    @classmethod
    def sample_docker_compose_services(cls) -> List[DockerComposeServiceConfig]:
        return [
            DockerComposeServiceConfig(
                service_name="ollama",
                image="ollama/ollama:latest",
                volumes=["$HOME/.ollama:/root/.ollama"],
                devices=["nvidia.com/gpu=all"],
                deploy={
                    "resources": {
                        "reservations": {
                            "devices": [{"driver": "nvidia", "capabilities": ["gpu"]}]
                        }
                    }
                },
                runtime="nvidia",
                ports=[f"{DEFAULT_OLLAMA_PORT}:{DEFAULT_OLLAMA_PORT}"],
                healthcheck={
                    "test": ["CMD", "curl", "-f", "http://ollama:11434"],
                    "interval": "10s",
                    "timeout": "5s",
                    "retries": 5,
                },
            ),
            DockerComposeServiceConfig(
                service_name="ollama-init",
                image="ollama/ollama",
                depends_on={"ollama": {"condition": "service_healthy"}},
                environment={
                    "OLLAMA_HOST": "ollama",
                    "OLLAMA_MODELS": "${OLLAMA_MODELS}",
                },
                volumes=["ollama_data:/root/.ollama"],
                entrypoint=(
                    'sh -c \'max_attempts=30;attempt=0;echo "Waiting for Ollama server...";'
                    "until curl -s http://ollama:11434 > /dev/null; do"
                    "attempt=$((attempt + 1));"
                    "if [ $attempt -ge $max_attempts ]; then"
                    'echo "Timeout waiting for Ollama server";'
                    "exit 1;"
                    "fi;"
                    'echo "Attempt $attempt: Server not ready yet...";'
                    "sleep 5;"
                    "done'"
                ),
            ),
        ]
