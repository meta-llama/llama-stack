# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

from llama_stack.schema_utils import json_schema_type


class JobStatus(Enum):
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


class BaseJob(BaseModel):
    id: str  # TODO: make it a UUID?

    # TODO: type will be defined as a Literal for each subclass; enforce its
    # presence with ABCMeta?

    # TODO: is there a way to provide default implementation that would extract
    # the result from the events?
    status: JobStatusDetails
    events: list[JobStatusDetails]

    artifacts: list[JobArtifact]
