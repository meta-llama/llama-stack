# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field
import boto3


@json_schema_type
class BedrockShieldConfig(BaseModel):
    """Configuration information for a guardrail that you want to use in the request."""

    aws_profile: Optional[str] = Field(
        default='default',
        description="The profile on the machine having valid aws credentials. This will ensure separation of creation to invocation",
    )



