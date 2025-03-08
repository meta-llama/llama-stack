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
        description="The NVIDIA API key, only needed of using the hosted service",
    )

    user_id: Optional[str] = Field(
        default_factory=lambda: os.getenv("NVIDIA_USER_ID", "llama-stack-user"),
        description="The NVIDIA user ID, only needed of using the hosted service",
    )

    dataset_namespace: Optional[str] = Field(
        default_factory=lambda: os.getenv("NVIDIA_DATASET_NAMESPACE", "default"),
        description="The NVIDIA dataset namespace, only needed of using the hosted service",
    )

    access_policies: Optional[dict] = Field(
        default_factory=lambda: os.getenv("NVIDIA_ACCESS_POLICIES", {}),
        description="The NVIDIA access policies, only needed of using the hosted service",
    )

    project_id: Optional[str] = Field(
        default_factory=lambda: os.getenv("NVIDIA_PROJECT_ID", "test-project"),
        description="The NVIDIA project ID, only needed of using the hosted service",
    )

    # ToDO: validate this, add default value
    customizer_url: str = Field(
        default_factory=lambda: os.getenv("NVIDIA_CUSTOMIZER_URL", "http://nemo.test"),
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

    output_model_dir: str = Field(
        default_factory=lambda: os.getenv("NVIDIA_OUTPUT_MODEL_DIR", "test-example-model@v1"),
        description="Directory to save the output model",
    )

    @classmethod
    def sample_run_config(cls, **kwargs) -> Dict[str, Any]:
        return {
            "api_key": "${env.NVIDIA_API_KEY:}",
            "user_id": "${env.NVIDIA_USER_ID:llama-stack-user}",
            "dataset_namespace": "${env.NVIDIA_DATASET_NAMESPACE:default}",
            "access_policies": "${env.NVIDIA_ACCESS_POLICIES:}",
            "project_id": "${env.NVIDIA_PROJECT_ID:test-project}",
            "customizer_url": "${env.NVIDIA_CUSTOMIZER_URL:}",
            "output_model_dir": "${env.NVIDIA_OUTPUT_MODEL_DIR:test-example-model@v1}",
        }
