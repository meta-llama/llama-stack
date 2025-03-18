# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class NvidiaPostTrainingConfig(BaseModel):
    """Configuration for NVIDIA Post Training implementation."""

    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("NVIDIA_API_KEY"),
        description="The NVIDIA API key.",
    )

    dataset_namespace: Optional[str] = Field(
        default_factory=lambda: os.getenv("NVIDIA_DATASET_NAMESPACE", "default"),
        description="The NVIDIA dataset namespace.",
    )

    project_id: Optional[str] = Field(
        default_factory=lambda: os.getenv("NVIDIA_PROJECT_ID", "test-example-model@v1"),
        description="The NVIDIA project ID.",
    )

    # ToDO: validate this, add default value
    customizer_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("NVIDIA_CUSTOMIZER_URL"),
        description="Base URL for the NeMo Customizer API",
    )

    timeout: int = Field(
        default=300,
        description="Timeout for the NVIDIA Post Training API",
    )

    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for the NVIDIA Post Training API",
    )

    # ToDo: validate this
    output_model_dir: str = Field(
        default_factory=lambda: os.getenv("NVIDIA_OUTPUT_MODEL_DIR", "test-example-model@v1"),
        description="Directory to save the output model",
    )

    @classmethod
    def sample_run_config(cls, **kwargs) -> Dict[str, Any]:
        return {
            "api_key": "${env.NVIDIA_API_KEY:}",
            "dataset_namespace": "${env.NVIDIA_DATASET_NAMESPACE:default}",
            "project_id": "${env.NVIDIA_PROJECT_ID:test-project}",
            "customizer_url": "${env.NVIDIA_CUSTOMIZER_URL:http://nemo.test}",
        }
