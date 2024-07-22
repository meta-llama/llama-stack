from enum import Enum
from typing import Dict, Optional

from llama_models.llama3_1.api.datatypes import URL

from pydantic import BaseModel

from strong_typing.schema import json_schema_type


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
    params: Optional[Dict[str, str]] = None
    headers: Optional[Dict[str, str]] = None
    body: Optional[Dict[str, str]] = None
