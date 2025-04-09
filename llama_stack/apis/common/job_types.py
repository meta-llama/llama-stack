# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from datetime import datetime, timezone
from enum import Enum, unique
from typing import Any

from pydantic import BaseModel, Field, computed_field

from llama_stack.schema_utils import json_schema_type


@unique
class JobStatus(Enum):
    unknown = "unknown"
    new = "new"
    scheduled = "scheduled"
    running = "running"
    paused = "paused"
    resuming = "resuming"
    cancelled = "cancelled"
    failed = "failed"
    completed = "completed"


@json_schema_type
class JobStatusDetails(BaseModel):
    status: JobStatus
    message: str | None = None
    timestamp: datetime


@json_schema_type
class JobArtifact(BaseModel):
    name: str

    # TODO: should it be a Literal / Enum?
    type: str

    # Any additional metadata the artifact may have
    # TODO: is Any the right type here? What happens when the caller passes a value without a __repr__?
    metadata: dict[str, Any] | None = None

    # TODO: enforce type to be a URI
    uri: str | None = None  # points to /files


def _get_job_status_details(status: JobStatus) -> JobStatusDetails:
    return JobStatusDetails(status=status, timestamp=datetime.now(timezone.utc))


class BaseJob(BaseModel):
    id: str  # TODO: make it a UUID?

    artifacts: list[JobArtifact] = Field(default_factory=list)
    events: list[JobStatusDetails] = Field(default_factory=lambda: [_get_job_status_details(JobStatus.new)])

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "type" not in cls.__annotations__:
            raise ValueError(f"Class {cls.__name__} must have a type field")

    @computed_field
    def status(self) -> JobStatus:
        return self.events[-1].status

    def update_status(self, value: JobStatus):
        self.events.append(_get_job_status_details(value))
