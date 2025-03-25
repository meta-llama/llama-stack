# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import List, Literal, Optional, Protocol

from pydantic import BaseModel

from llama_stack.apis.common.job_types import BaseJob
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
class SyntheticDataGenerationJob(BaseJob, BaseModel):
    type: Literal["synthetic-data-generation"] = "synthetic-data-generation"


@json_schema_type
class ListSyntheticDataGenerationJobsResponse(BaseModel):
    items: list[SyntheticDataGenerationJob]


class SyntheticDataGeneration(Protocol):
    @webmethod(route="/synthetic-data-generation/generate")
    def synthetic_data_generate(
        self,
        dialogs: List[Message],
        filtering_function: FilteringFunction = FilteringFunction.none,
        model: Optional[str] = None,
    ) -> SyntheticDataGenerationJob: ...

    # CRUD operations on running jobs
    @webmethod(route="/synthetic-data-generation/jobs/{job_id:path}", method="GET")
    async def get_synthetic_data_generation_job(self) -> SyntheticDataGenerationJob: ...

    @webmethod(route="/synthetic-data-generation/jobs", method="GET")
    async def list_synthetic_data_generation_jobs(self) -> ListSyntheticDataGenerationJobsResponse: ...

    @webmethod(route="/synthetic-data-generation/jobs/{job_id:path}", method="POST")
    async def update_synthetic_data_generation_job(
        self, job: SyntheticDataGenerationJob
    ) -> SyntheticDataGenerationJob: ...

    @webmethod(route="/synthetic-data-generation/job/{job_id:path}", method="DELETE")
    async def delete_synthetic_data_generation_job(self, job_id: str) -> None: ...

    # Note: pause/resume/cancel are achieved as follows:
    # - POST with status=paused
    # - POST with status=resuming
    # - POST with status=cancelled
