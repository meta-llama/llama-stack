# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel

from llama_stack.apis.common.job_types import JobStatus
from llama_stack.apis.inference import Message
from llama_stack.schema_utils import json_schema_type, webmethod


class FilteringFunction(Enum):
    """The type of filtering function."""

    none = "none"
    random = "random"
    top_k = "top_k"
    top_p = "top_p"
    top_k_top_p = "top_k_top_p"
    sigmoid = "sigmoid"


@json_schema_type
class SyntheticDataGenerationRequest(BaseModel):
    """Request to generate synthetic data. A small batch of prompts and a filtering function"""

    dialogs: List[Message]
    filtering_function: FilteringFunction = FilteringFunction.none
    model: Optional[str] = None


@json_schema_type
class SyntheticDataGenerationJob(BaseModel):
    job_uuid: str


@json_schema_type
class SyntheticDataGenerationJobStatusResponse(BaseModel):
    """Status of a synthetic data generation job."""

    job_uuid: str
    status: JobStatus

    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    synthetic_data: List[Dict[str, Any]]
    statistics: Optional[Dict[str, Any]] = None


class ListSyntheticDataGenerationJobsResponse(BaseModel):
    data: List[SyntheticDataGenerationJob]


@json_schema_type
class SyntheticDataGenerationJobArtifactsResponse(BaseModel):
    job_uuid: str

    synthetic_data: List[Dict[str, Any]]
    statistics: Optional[Dict[str, Any]] = None


class SyntheticDataGeneration(Protocol):
    @webmethod(route="/synthetic-data-generation/generate", method="POST")
    def synthetic_data_generate(
        self,
        dialogs: List[Message],
        filtering_function: FilteringFunction = FilteringFunction.none,
        model: Optional[str] = None,
    ) -> SyntheticDataGenerationJob: ...

    @webmethod(route="/synthetic-data-generation/jobs", method="GET")
    async def get_synthetic_data_generation_jobs(self) -> ListSyntheticDataGenerationJobsResponse: ...

    @webmethod(route="/synthetic-data-generation/job/status", method="GET")
    async def get_synthetic_data_generation_job_status(
        self, job_uuid: str
    ) -> Optional[SyntheticDataGenerationJobStatusResponse]: ...

    @webmethod(route="/synthetic-data-generation/job/cancel", method="POST")
    async def cancel_synthetic_data_generation_job(self, job_uuid: str) -> None: ...

    @webmethod(route="/synthetic-data-generation/job/artifacts", method="GET")
    async def get_synthetic_data_generation_job_artifacts(
        self, job_uuid: str
    ) -> Optional[SyntheticDataGenerationJobArtifactsResponse]: ...
