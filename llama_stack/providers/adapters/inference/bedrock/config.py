# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import *  # noqa: F403

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field


@json_schema_type
class BedrockConfig(BaseModel):
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
    profile_name: Optional[str] = Field(
        default=None,
        description="The profile name that contains credentials to use."
        "Default use environment variable: AWS_PROFILE",
    )
    total_max_attempts: Optional[int] = Field(
        default=None,
        description="An integer representing the maximum number of attempts that will be made for a single request, "
        "including the initial attempt. Default use environment variable: AWS_MAX_ATTEMPTS",
    )
    retry_mode: Optional[str] = Field(
        default=None,
        description="A string representing the type of retries Boto3 will perform."
        "Default use environment variable: AWS_RETRY_MODE",
    )
    connect_timeout: Optional[float] = Field(
        default=60,
        description="The time in seconds till a timeout exception is thrown when attempting to make a connection. "
        "The default is 60 seconds.",
    )
    read_timeout: Optional[float] = Field(
        default=60,
        description="The time in seconds till a timeout exception is thrown when attempting to read from a connection."
        "The default is 60 seconds.",
    )
