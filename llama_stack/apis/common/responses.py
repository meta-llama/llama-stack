# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, TypeVar

from pydantic import BaseModel

from llama_stack.schema_utils import json_schema_type

T = TypeVar("T")


@json_schema_type
class PaginatedResponse(BaseModel):
    """A generic paginated response that can be used across different APIs.

    :param data: The list of items for the current page
    :param total: Total number of items across all pages
    :param page: Current page number (1-based indexing)
    :param size: Number of items per page
    :param pages: Total number of pages available
    """

    data: List[Dict[str, Any]]
    total: int
    page: int
    size: int
    pages: int
