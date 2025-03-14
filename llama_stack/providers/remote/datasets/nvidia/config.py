# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import warnings
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class NvidiaDatasetConfig(BaseModel):
    """Configuration for NVIDIA Dataset implementation."""

    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("NVIDIA_API_KEY"),
        description="The NVIDIA API key.",
    )

    dataset_namespace: Optional[str] = Field(
        default_factory=lambda: os.getenv("NVIDIA_DATASET_NAMESPACE", "default"),
        description="The NVIDIA dataset namespace.",
    )

    access_policies: Optional[dict] = Field(
        default_factory=lambda: os.getenv("NVIDIA_ACCESS_POLICIES", {}),
        description="The NVIDIA access policies.",
    )

    project_id: Optional[str] = Field(
        default_factory=lambda: os.getenv("NVIDIA_PROJECT_ID", "test-project"),
        description="The NVIDIA project ID.",
    )

    datasets_url: str = Field(
        default_factory=lambda: os.getenv("NVIDIA_DATASETS_URL", "http://nemo.test"),
        description="Base URL for the NeMo Dataset API",
    )

    # warning for default values
    def __post_init__(self):
        default_values = []
        if os.getenv("NVIDIA_PROJECT_ID") is None:
            default_values.append("project_id='test-project'")
        if os.getenv("NVIDIA_DATASET_NAMESPACE") is None:
            default_values.append("dataset_namespace='default'")
        if os.getenv("NVIDIA_ACCESS_POLICIES") is None:
            default_values.append("access_policies='{}'")
        if os.getenv("NVIDIA_DATASETS_URL") is None:
            default_values.append("datasets_url='http://nemo.test'")

        if default_values:
            warnings.warn(
                f"Using default values: {', '.join(default_values)}. \
                          Please set the environment variables to avoid this default behavior.",
                stacklevel=2,
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
