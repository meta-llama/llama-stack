# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import sys
from enum import Enum

from pydantic import BaseModel, Field

# TODO: use enum.StrEnum when we drop support for python 3.10
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, Enum):
        """Backport of StrEnum for Python 3.10 and below."""

        pass


class ResourceType(StrEnum):
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

    provider_resource_id: str | None = Field(
        default=None,
        description="Unique identifier for this resource in the provider",
    )

    provider_id: str = Field(description="ID of the provider that owns this resource")

    type: ResourceType = Field(description="Type of resource (e.g. 'model', 'shield', 'vector_db', etc.)")
