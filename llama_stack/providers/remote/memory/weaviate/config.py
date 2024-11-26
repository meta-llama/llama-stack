# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel


class WeaviateRequestProviderData(BaseModel):
    weaviate_api_key: str
    weaviate_cluster_url: str


class WeaviateConfig(BaseModel):
    pass
