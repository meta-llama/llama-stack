# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from typing import Optional

from pydantic import BaseModel, Field


class NvidiaPostTrainingConfig(BaseModel):
    """Configuration for NVIDIA Post Training implementation."""

    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("NVIDIA_API_KEY"),
        description="The NVIDIA API key, only needed of using the hosted service",
    )

    user_id: Optional[str] = Field(
        default_factory=lambda: os.getenv("NVIDIA_USER_ID"),
        description="The NVIDIA user ID, only needed of using the hosted service",
    )

    customizer_url: str = Field(
        default_factory=lambda: os.getenv("NVIDIA_CUSTOMIZER_URL"),
        description="Base URL for the NeMo Customizer API",
    )
