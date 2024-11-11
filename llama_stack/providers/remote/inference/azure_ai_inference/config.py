# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import *  # noqa: F403

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field


@json_schema_type
class AzureAIInferenceConfig(BaseModel):
    endpoint: str = Field(
        default=None,
        description="The endpoint URL where the model(s) is/are deployed.",
    )
    credential: Optional[str] = Field(
        default=None,
        description="The secret to access the model. If None, then `DefaultAzureCredential` is attempted.",
    )
    api_version: Optional[str] = Field(
        default=None,
        description="The API version to use in the endpoint. Indicating None will use the default version in the "
        "`azure-ai-inference` package. Default use environment variable: AZURE_AI_API_VERSION",
    )