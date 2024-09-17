# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Dict, Optional

from llama_models.llama3.api.datatypes import URL

from llama_models.schema_utils import json_schema_type

from pydantic import BaseModel


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
