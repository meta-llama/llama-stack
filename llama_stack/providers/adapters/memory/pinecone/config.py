# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel


class PineconeRequestProviderData(BaseModel):
    pinecone_api_key: str


class PineconeConfig(BaseModel):
    dimension: int = 384
    cloud: str = "aws"
    region: str = "us-east-1"
