# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from llama_stack.apis.common.job_types import CommonJobFields, JobType
from llama_stack.apis.scoring_functions import ScoringFn
from llama_stack.schema_utils import json_schema_type, webmethod

# mapping of metric to value
ScoringResultRow = Dict[str, Any]


@json_schema_type
class ScoringResult(BaseModel):
    """
    A scoring result for a single row.

    :param score_rows: The scoring result for each row. Each row is a map of column name to value.
    :param aggregated_results: Map of metric name to aggregated value
    """

    score_rows: List[ScoringResultRow]
    # aggregated metrics to value
    aggregated_results: Dict[str, Any]


@json_schema_type
class ScoreBatchResponse(BaseModel):
    dataset_id: Optional[str] = None
    results: Dict[str, ScoringResult]


@json_schema_type
class ScoreResponse(BaseModel):
    """
    The response from scoring.

    :param results: A map of scoring function name to ScoringResult.
    """

    # each key in the dict is a scoring function name
    results: Dict[str, ScoringResult]


@json_schema_type
class ScoringJob(CommonJobFields):
    """
    A scoring job.

    :param type: The type of the job.
    :param result_files: The file ids of the scoring results.
    :param result_datasets: The ids of the datasets containing the scoring results.
    :param dataset_id: The id of the dataset used for scoring.
    :param scoring_fn_ids: The ids of the scoring functions used.
    """

    type: JobType = JobType.scoring.value

    result_files: List[str] = Field(
        description="The file ids of the scoring results.",
        default_factory=list,
    )
    result_datasets: List[str] = Field(
        description="The ids of the datasets containing the scoring results.",
        default_factory=list,
    )

    # how the job is created
    dataset_id: str = Field(description="The id of the dataset used for scoring.")
    scoring_fn_ids: List[str] = Field(
        description="The ids of the scoring functions used.",
        default_factory=list,
    )


class ScoringFunctionStore(Protocol):
    def get_scoring_function(self, scoring_fn_id: str) -> ScoringFn: ...


@runtime_checkable
class Scoring(Protocol):
    scoring_function_store: ScoringFunctionStore

    @webmethod(route="/scoring/jobs", method="POST")
    async def score_dataset(
        self,
        dataset_id: str,
        scoring_fn_ids: List[str],
    ) -> ScoringJob: ...

    @webmethod(route="/scoring/rows", method="POST")
    async def score(
        self,
        dataset_rows: List[Dict[str, Any]],
        scoring_fn_ids: List[str],
    ) -> ScoreResponse:
        """Score a list of rows.

        :param dataset_rows: The rows to score.
        :param scoring_fn_ids: The scoring function ids to use for the scoring.
        :return: ScoreResponse object containing rows and aggregated results
        """
        ...

    @webmethod(route="/scoring/jobs", method="GET")
    async def list_scoring_jobs(self) -> List[ScoringJob]:
        """List all scoring jobs.

        :return: A list of scoring jobs.
        """
        ...

    @webmethod(route="/scoring/jobs/{job_id}", method="GET")
    async def get_scoring_job(self, job_id: str) -> Optional[ScoringJob]:
        """Get a job by id.

        :param job_id: The id of the job to get.
        :return: The job.
        """
        ...

    @webmethod(route="/scoring/jobs/{job_id}", method="DELETE")
    async def delete_scoring_job(self, job_id: str) -> Optional[ScoringJob]:
        """Delete a job.

        :param job_id: The id of the job to delete.
        """
        ...

    @webmethod(route="/scoring/jobs/{job_id}/cancel", method="POST")
    async def cancel_scoring_job(self, job_id: str) -> Optional[ScoringJob]:
        """Cancel a job.

        :param job_id: The id of the job to cancel.
        """
        ...
