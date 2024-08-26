# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from enum import Enum

from typing import Any, Dict, List, Optional, Protocol, Union

from llama_models.schema_utils import json_schema_type, webmethod

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


@json_schema_type
class CreateExperimentRequest(BaseModel):
    name: str
    metadata: Optional[Dict[str, Any]] = None


@json_schema_type
class UpdateExperimentRequest(BaseModel):
    experiment_id: str
    status: Optional[ExperimentStatus] = None
    metadata: Optional[Dict[str, Any]] = None


@json_schema_type
class CreateRunRequest(BaseModel):
    experiment_id: str
    metadata: Optional[Dict[str, Any]] = None


@json_schema_type
class UpdateRunRequest(BaseModel):
    run_id: str
    status: Optional[str] = None
    ended_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


@json_schema_type
class LogMetricsRequest(BaseModel):
    run_id: str
    metrics: List[Metric]


@json_schema_type
class LogMessagesRequest(BaseModel):
    logs: List[Log]
    run_id: Optional[str] = None


@json_schema_type
class UploadArtifactRequest(BaseModel):
    experiment_id: str
    name: str
    artifact_type: str
    content: bytes
    metadata: Optional[Dict[str, Any]] = None


@json_schema_type
class LogSearchRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None


class Observability(Protocol):
    @webmethod(route="/experiments/create")
    def create_experiment(self, request: CreateExperimentRequest) -> Experiment: ...

    @webmethod(route="/experiments/list")
    def list_experiments(self) -> List[Experiment]: ...

    @webmethod(route="/experiments/get")
    def get_experiment(self, experiment_id: str) -> Experiment: ...

    @webmethod(route="/experiments/update")
    def update_experiment(self, request: UpdateExperimentRequest) -> Experiment: ...

    @webmethod(route="/experiments/create_run")
    def create_run(self, request: CreateRunRequest) -> Run: ...

    @webmethod(route="/runs/update")
    def update_run(self, request: UpdateRunRequest) -> Run: ...

    @webmethod(route="/runs/log_metrics")
    def log_metrics(self, request: LogMetricsRequest) -> None: ...

    @webmethod(route="/runs/metrics", method="GET")
    def get_metrics(self, run_id: str) -> List[Metric]: ...

    @webmethod(route="/logging/log_messages")
    def log_messages(self, request: LogMessagesRequest) -> None: ...

    @webmethod(route="/logging/get_logs")
    def get_logs(self, request: LogSearchRequest) -> List[Log]: ...

    @webmethod(route="/experiments/artifacts/upload")
    def upload_artifact(self, request: UploadArtifactRequest) -> Artifact: ...

    @webmethod(route="/experiments/artifacts/get")
    def list_artifacts(self, experiment_id: str) -> List[Artifact]: ...

    @webmethod(route="/artifacts/get")
    def get_artifact(self, artifact_id: str) -> Artifact: ...
