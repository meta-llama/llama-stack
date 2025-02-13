# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

from llama_models.schema_utils import json_schema_type, webmethod
from pydantic import BaseModel, Field

from llama_stack.apis.resource import Resource, ResourceType


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
    def task_id(self) -> str:
        return self.identifier

    @property
    def provider_benchmark_id(self) -> str:
        return self.provider_resource_id


class BenchmarkInput(CommonBenchmarkFields, BaseModel):
    task_id: str
    provider_id: Optional[str] = None
    provider_benchmark_id: Optional[str] = None


class ListBenchmarksResponse(BaseModel):
    data: List[Benchmark]


@runtime_checkable
class Benchmarks(Protocol):
    @webmethod(route="/eval/tasks", method="GET")
    async def list_benchmarks(self) -> ListBenchmarksResponse: ...

    @webmethod(route="/eval/tasks/{task_id}", method="GET")
    async def get_benchmark(
        self,
        task_id: str,
    ) -> Optional[Benchmark]: ...

    @webmethod(route="/eval/tasks", method="POST")
    async def register_benchmark(
        self,
        task_id: str,
        dataset_id: str,
        scoring_functions: List[str],
        provider_benchmark_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: ...

    @webmethod(route="/eval-tasks", method="GET")
    async def DEPRECATED_list_benchmarks(self) -> ListBenchmarksResponse: ...

    @webmethod(route="/eval-tasks/{benchmark_id}", method="GET")
    async def DEPRECATED_get_benchmark(
        self,
        benchmark_id: str,
    ) -> Optional[Benchmark]: ...

    @webmethod(route="/eval-tasks", method="POST")
    async def DEPRECATED_register_benchmark(
        self,
        benchmark_id: str,
        dataset_id: str,
        scoring_functions: List[str],
        provider_benchmark_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: ...
