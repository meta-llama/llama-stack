# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.apis.scoring_functions import ScoringFnParams
from llama_stack.schema_utils import json_schema_type, webmethod


class CommonBenchmarkFields(BaseModel):
    """
    :param dataset_id: The ID of the dataset to used to run the benchmark.
    :param scoring_functions: The scoring functions with parameters to use for this benchmark.
    :param metadata: Metadata for this benchmark for additional descriptions.
    """

    dataset_id: str
    scoring_fn_ids: List[str]
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata for this benchmark",
    )


@json_schema_type
class Benchmark(CommonBenchmarkFields, Resource):
    type: Literal[ResourceType.benchmark.value] = ResourceType.benchmark.value

    @property
    def benchmark_id(self) -> str:
        return self.identifier

    @property
    def provider_benchmark_id(self) -> str:
        return self.provider_resource_id


class BenchmarkInput(CommonBenchmarkFields, BaseModel):
    benchmark_id: str
    provider_id: Optional[str] = None
    provider_benchmark_id: Optional[str] = None


class ListBenchmarksResponse(BaseModel):
    data: List[Benchmark]


@runtime_checkable
class Benchmarks(Protocol):
    @webmethod(route="/eval/benchmarks", method="GET")
    async def list_benchmarks(self) -> ListBenchmarksResponse: ...

    @webmethod(route="/eval/benchmarks/{benchmark_id}", method="GET")
    async def get_benchmark(
        self,
        benchmark_id: str,
    ) -> Optional[Benchmark]: ...

    @webmethod(route="/eval/benchmarks", method="POST")
    async def register_benchmark(
        self,
        dataset_id: str,
        scoring_fn_ids: List[str],
        benchmark_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Benchmark:
        """
        Register a new benchmark.

        :param dataset_id: The ID of the dataset to used to run the benchmark.
        :param scoring_fn_ids: List of scoring function ids to use for this benchmark.
        :param benchmark_id: (Optional) The ID of the benchmark to register. If not provided, a random ID will be generated.
        :param metadata: (Optional) Metadata for this benchmark for additional descriptions.
        """
        ...
