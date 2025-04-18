# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, Field


class S3ImplConfig(BaseModel):
    """Configuration for S3 file storage provider."""

    aws_access_key_id: str = Field(description="AWS access key ID")
    aws_secret_access_key: str = Field(description="AWS secret access key")
    region_name: str | None = Field(default=None, description="AWS region name")
    endpoint_url: str | None = Field(default=None, description="Optional endpoint URL for S3 compatible services")
    bucket_name: str | None = Field(default=None, description="Default S3 bucket name")
    verify_tls: bool = Field(default=True, description="Verify TLS certificates")
