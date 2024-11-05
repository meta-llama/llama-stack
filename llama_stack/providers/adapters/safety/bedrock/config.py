# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, Field
from typing import Optional

class BedrockSafetyConfig(BaseModel):
    """Configuration information for a guardrail that you want to use in the request."""

    aws_access_key_id: Optional[str] = Field(
        default=None,
        description="The AWS access key to use. Default use environment variable: AWS_ACCESS_KEY_ID",
    )
    aws_secret_access_key: Optional[str] = Field(
        default=None,
        description="The AWS secret access key to use. Default use environment variable: AWS_SECRET_ACCESS_KEY",
    )
    aws_session_token: Optional[str] = Field(
        default=None,
        description="The AWS session token to use. Default use environment variable: AWS_SESSION_TOKEN",
    )
    region_name: Optional[str] = Field(
        default=None,
        description="The default AWS Region to use, for example, us-west-1 or us-west-2."
        "Default use environment variable: AWS_DEFAULT_REGION",
    )
    profile_name: str = Field(
        default="default",
        description="The profile on the machine having valid aws credentials. This will ensure separation of creation to invocation",
    )
