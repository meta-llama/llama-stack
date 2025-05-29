# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

import requests
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from llama_stack.apis.inference import Message
from llama_stack.apis.synthetic_data_generation import (
    FilteringFunction,
    SyntheticDataGeneration,
    SyntheticDataGenerationResponse,
)


class SyntheticDataKitConfig(BaseModel):
    """Configuration for the Synthetic Data Kit provider"""
    llm: Dict[str, Any] = Field(
        default_factory=lambda: {
            "provider": "vllm",
            "model": "meta-llama/Llama-3.2-3B-Instruct",
        }
    )
    vllm: Dict[str, Any] = Field(
        default_factory=lambda: {
            "api_base": "http://localhost:8000/v1",
        }
    )
    generation: Dict[str, Any] = Field(
        default_factory=lambda: {
            "temperature": 0.7,
            "chunk_size": 4000,
            "num_pairs": 25,
        }
    )
    curate: Dict[str, Any] = Field(
        default_factory=lambda: {
            "threshold": 7.0,
            "batch_size": 8,
        }
    )

    @classmethod
    def sample_run_config(cls) -> "SyntheticDataKitConfig":
        """Create a sample configuration for testing"""
        return cls()


class SyntheticDataKitProvider(SyntheticDataGeneration):
    def __init__(self, config: SyntheticDataKitConfig):
        self.config = config
        self._validate_connection()

    def _validate_connection(self) -> None:
        """Validate connection to vLLM server"""
        try:
            response = requests.get(f"http://localhost:{self.config.vllm['port']}/health")
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to vLLM server: {e}") from e

    def synthetic_data_generate(
        self,
        dialogs: list[Message],
        filtering_function: FilteringFunction = FilteringFunction.none,
        model: str | None = None,
    ) -> SyntheticDataGenerationResponse:
        # Convert dialogs to SDK format
        formatted_dialogs = [{"role": dialog.role, "content": dialog.content} for dialog in dialogs]

        payload = {
            "dialogs": formatted_dialogs,
            "filtering_function": filtering_function.value,
            "model": model or self.config.llm["model"],
            "generation": self.config.generation,
            "curate": self.config.curate if filtering_function != FilteringFunction.none else None,
        }

        try:
            response = requests.post(
                f"http://localhost:{self.config.vllm['port']}/v1/synthetic-data-generation/generate",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            return SyntheticDataGenerationResponse(
                synthetic_data=result.get("synthetic_data", []),
                statistics=result.get("statistics"),
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Synthetic data generation failed: {e}") from e
