# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel

from llama_stack.apis.common.content_types import URL
from llama_stack.schema_utils import json_schema_type


@json_schema_type
class RestAPIMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


@json_schema_type
class RestAPIExecutionConfig(BaseModel):
    url: URL
    method: RestAPIMethod
    params: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, Any]] = None
    body: Optional[Dict[str, Any]] = None
