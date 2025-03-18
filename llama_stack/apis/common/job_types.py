# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from llama_stack.schema_utils import json_schema_type


@json_schema_type
class Job(BaseModel):
    # NOTE: this will be DEPRECATED in favour of CommonJobFields
    job_id: str


class JobType(Enum):
    batch_inference = "batch_inference"
    evaluation = "evaluation"
    finetuning = "finetuning"


class JobStatus(Enum):
    completed = "completed"
    in_progress = "in_progress"
    failed = "failed"
    scheduled = "scheduled"
    cancelled = "cancelled"


@json_schema_type
class CommonJobFields(BaseModel):
    """Common fields for all jobs.
    :param id: The ID of the job.
    :param status: The status of the job.
    :param created_at: The time the job was created.
    :param completed_at: The time the job completed.
    :param error: If status of the job is failed, this will contain the error message.
    """

    id: str
    status: JobStatus
    created_at: datetime
    completed_at: datetime | None = None
    error: str | None = None
