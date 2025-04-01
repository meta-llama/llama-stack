# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, Field

from llama_stack.providers.utils.kvstore import KVStoreConfig
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig


class S3FilesImplConfig(BaseModel):
    """Configuration for S3 file storage provider."""

    aws_access_key_id: str = Field(description="AWS access key ID")
    aws_secret_access_key: str = Field(description="AWS secret access key")
    region_name: str | None = Field(default=None, description="AWS region name")
    endpoint_url: str | None = Field(default=None, description="Optional endpoint URL for S3 compatible services")
    bucket_name: str | None = Field(default=None, description="Default S3 bucket name")
    verify_tls: bool = Field(default=True, description="Verify TLS certificates")
    persistent_store: KVStoreConfig

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict:
        return {
            "aws_access_key_id": "your-access-key-id",
            "aws_secret_access_key": "your-secret-access-key",
            "region_name": "us-west-2",
            "endpoint_url": None,
            "bucket_name": "your-bucket-name",
            "verify_tls": True,
            "persistence_store": SqliteKVStoreConfig.sample_run_config(
                __distro_dir__=__distro_dir__,
                db_name="files_s3_store.db",
            ),
        }
