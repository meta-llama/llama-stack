# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum

from pydantic import BaseModel, Field


class ResourceType(Enum):
    model = "model"
    shield = "shield"
    vector_db = "vector_db"
    dataset = "dataset"
    scoring_function = "scoring_function"
    benchmark = "benchmark"
    tool = "tool"
    tool_group = "tool_group"


class Resource(BaseModel):
    """Base class for all Llama Stack resources"""

    identifier: str = Field(description="Unique identifier for this resource in llama stack")

    provider_resource_id: str = Field(
        description="Unique identifier for this resource in the provider",
        default=None,
    )

    provider_id: str = Field(description="ID of the provider that owns this resource")

    type: ResourceType = Field(description="Type of resource (e.g. 'model', 'shield', 'vector_db', etc.)")
