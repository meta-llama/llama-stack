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
    dataset_id: str
    scoring_functions: List[str]
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata for this evaluation task",
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
    ) -> Benchmark: ...

    @webmethod(route="/eval/benchmarks", method="POST")
    async def register_benchmark(
        self,
        benchmark_id: str,
        dataset_id: str,
        scoring_functions: List[str],
        provider_benchmark_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: ...
