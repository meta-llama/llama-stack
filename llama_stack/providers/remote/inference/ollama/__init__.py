# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from llama_stack.distribution.datatypes import RemoteProviderConfig
from llama_stack.providers.utils.docker.service_config import DockerComposeServiceConfig


DEFAULT_OLLAMA_PORT = 11434


class OllamaImplConfig(RemoteProviderConfig):
    port: int = DEFAULT_OLLAMA_PORT

    @classmethod
    def sample_docker_compose_config(cls) -> Optional[DockerComposeServiceConfig]:
        return DockerComposeServiceConfig(
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
        )


async def get_adapter_impl(config: RemoteProviderConfig, _deps):
    from .ollama import OllamaInferenceAdapter

    impl = OllamaInferenceAdapter(config.url)
    await impl.initialize()
    return impl
