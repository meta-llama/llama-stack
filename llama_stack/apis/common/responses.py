# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List

from pydantic import BaseModel

from llama_stack.schema_utils import json_schema_type


@json_schema_type
class PaginatedResponse(BaseModel):
    """A generic paginated response that follows a simple format.

    :param data: The list of items for the current page
    :param has_more: Whether there are more items available after this set
    """

    data: List[Dict[str, Any]]
    has_more: bool
