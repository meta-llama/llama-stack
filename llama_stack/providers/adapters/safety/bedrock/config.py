# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, Field


class BedrockSafetyConfig(BaseModel):
    """Configuration information for a guardrail that you want to use in the request."""

    aws_profile: str = Field(
        default="default",
        description="The profile on the machine having valid aws credentials. This will ensure separation of creation to invocation",
    )
