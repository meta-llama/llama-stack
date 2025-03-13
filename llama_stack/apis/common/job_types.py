# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel

from llama_stack.schema_utils import json_schema_type


class JobStatus(Enum):
    completed = "completed"
    in_progress = "in_progress"
    failed = "failed"
    scheduled = "scheduled"
    cancelled = "cancelled"


class JobType(Enum):
    batch_inference = "batch_inference"
    scoring = "scoring"
    evaluation = "evaluation"
    post_training = "post_training"


@json_schema_type
class CommonJobFields(BaseModel):
    """Common fields for all jobs.
    :param id: The ID of the job.
    :param status: The status of the job.
    :param created_at: The time the job was created.
    :param finished_at: The time the job finished.
    :param error: If status of the job is failed, this will contain the error message.
    """

    id: str
    status: JobStatus
    created_at: datetime
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
