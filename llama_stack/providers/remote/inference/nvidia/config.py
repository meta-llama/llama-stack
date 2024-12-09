# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from typing import Optional

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field


@json_schema_type
class NVIDIAConfig(BaseModel):
    """
    Configuration for the NVIDIA NIM inference endpoint.

    Attributes:
        url (str): A base url for accessing the NVIDIA NIM, e.g. http://localhost:8000
        api_key (str): The access key for the hosted NIM endpoints

    There are two ways to access NVIDIA NIMs -
     0. Hosted: Preview APIs hosted at https://integrate.api.nvidia.com
     1. Self-hosted: You can run NVIDIA NIMs on your own infrastructure

    By default the configuration is set to use the hosted APIs. This requires
    an API key which can be obtained from https://ngc.nvidia.com/.

    By default the configuration will attempt to read the NVIDIA_API_KEY environment
    variable to set the api_key. Please do not put your API key in code.

    If you are using a self-hosted NVIDIA NIM, you can set the url to the
    URL of your running NVIDIA NIM and do not need to set the api_key.
    """

    url: str = Field(
        default_factory=lambda: os.getenv(
            "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com"
        ),
        description="A base url for accessing the NVIDIA NIM",
    )
    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("NVIDIA_API_KEY"),
        description="The NVIDIA API key, only needed of using the hosted service",
    )
    timeout: int = Field(
        default=60,
        description="Timeout for the HTTP requests",
    )