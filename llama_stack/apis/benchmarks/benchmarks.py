# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.schema_utils import json_schema_type, webmethod


class CommonBenchmarkFields(BaseModel):
    """
    :param dataset_id: The ID of the dataset to used to run the benchmark.
    :param grader_ids: The grader ids to use for this benchmark.
    :param metadata: Metadata for this benchmark for additional descriptions.
    """

    dataset_id: str
    grader_ids: List[str]
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
    @webmethod(route="/benchmarks", method="POST")
    async def register_benchmark(
        self,
        dataset_id: str,
        grader_ids: List[str],
        benchmark_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Benchmark:
        """
        Register a new benchmark. A benchmark consists of a dataset id and a list of grader ids.

        :param dataset_id: The ID of the dataset to be used to run the benchmark.
        :param grader_ids: List of grader ids to use for this benchmark.
        :param benchmark_id: (Optional) The ID of the benchmark to register. If not provided, an ID will be generated.
        :param metadata: (Optional) Metadata for this benchmark for additional descriptions.
        """
        ...

    @webmethod(route="/benchmarks", method="GET")
    async def list_benchmarks(self) -> ListBenchmarksResponse:
        """
        List all benchmarks.
        """
        ...

    @webmethod(route="/benchmarks/{benchmark_id}", method="GET")
    async def get_benchmark(
        self,
        benchmark_id: str,
    ) -> Benchmark:
        """
        Get a benchmark by ID.

        :param benchmark_id: The ID of the benchmark to get.
        """
        ...

    @webmethod(route="/benchmarks/{benchmark_id}", method="DELETE")
    async def unregister_benchmark(self, benchmark_id: str) -> None:
        """
        Unregister a benchmark by ID.
        """
        ...
