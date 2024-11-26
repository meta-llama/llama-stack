# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from enum import Enum

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel


@json_schema_type
class Job(BaseModel):
    job_id: str


@json_schema_type
class JobStatus(Enum):
    completed = "completed"
    in_progress = "in_progress"
