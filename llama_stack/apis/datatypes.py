# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Optional

from pydantic import BaseModel

from llama_stack.schema_utils import json_schema_type


@json_schema_type
class Api(Enum):
    providers = "providers"
    inference = "inference"
    safety = "safety"
    agents = "agents"
    vector_io = "vector_io"
    datasetio = "datasetio"
    scoring = "scoring"
    eval = "eval"
    post_training = "post_training"
    tool_runtime = "tool_runtime"

    telemetry = "telemetry"

    models = "models"
    shields = "shields"
    vector_dbs = "vector_dbs"
    datasets = "datasets"
    scoring_functions = "scoring_functions"
    benchmarks = "benchmarks"
    tool_groups = "tool_groups"
    files = "files"

    # built-in API
    inspect = "inspect"


@json_schema_type
class Error(BaseModel):
    """
    Error response from the API. Roughly follows RFC 7807.

    :param status: HTTP status code
    :param title: Error title, a short summary of the error which is invariant for an error type
    :param detail: Error detail, a longer human-readable description of the error
    :param instance: (Optional) A URL which can be used to retrieve more information about the specific occurrence of the error
    """

    status: int
    title: str
    detail: str
    instance: Optional[str] = None
