# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from enum import Enum

from typing import Any, Dict, Optional, Union

from llama_models.schema_utils import json_schema_type

from pydantic import BaseModel


@json_schema_type
class ExperimentStatus(Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@json_schema_type
class Experiment(BaseModel):
    id: str
    name: str
    status: ExperimentStatus
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]


@json_schema_type
class Run(BaseModel):
    id: str
    experiment_id: str
    status: str
    started_at: datetime
    ended_at: Optional[datetime]
    metadata: Dict[str, Any]


@json_schema_type
class Metric(BaseModel):
    name: str
    value: Union[float, int, str, bool]
    timestamp: datetime
    run_id: str


@json_schema_type
class Log(BaseModel):
    message: str
    level: str
    timestamp: datetime
    additional_info: Dict[str, Any]


@json_schema_type
class ArtifactType(Enum):
    MODEL = "model"
    DATASET = "dataset"
    CHECKPOINT = "checkpoint"
    PLOT = "plot"
    METRIC = "metric"
    CONFIG = "config"
    CODE = "code"
    OTHER = "other"


@json_schema_type
class Artifact(BaseModel):
    id: str
    name: str
    type: ArtifactType
    size: int
    created_at: datetime
    metadata: Dict[str, Any]
