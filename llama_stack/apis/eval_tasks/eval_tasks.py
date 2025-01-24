# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

from llama_models.schema_utils import json_schema_type, webmethod
from pydantic import BaseModel, Field

from llama_stack.apis.resource import Resource, ResourceType


class CommonEvalTaskFields(BaseModel):
    dataset_id: str
    scoring_functions: List[str]
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata for this evaluation task",
    )


@json_schema_type
class EvalTask(CommonEvalTaskFields, Resource):
    type: Literal[ResourceType.eval_task.value] = ResourceType.eval_task.value

    @property
    def eval_task_id(self) -> str:
        return self.identifier

    @property
    def provider_eval_task_id(self) -> str:
        return self.provider_resource_id


class EvalTaskInput(CommonEvalTaskFields, BaseModel):
    eval_task_id: str
    provider_id: Optional[str] = None
    provider_eval_task_id: Optional[str] = None


class ListEvalTasksResponse(BaseModel):
    data: List[EvalTask]


@runtime_checkable
class EvalTasks(Protocol):
    @webmethod(route="/eval-tasks", method="GET")
    async def list_eval_tasks(self) -> ListEvalTasksResponse: ...

    @webmethod(route="/eval-tasks/{eval_task_id}", method="GET")
    async def get_eval_task(
        self,
        eval_task_id: str,
    ) -> Optional[EvalTask]: ...

    @webmethod(route="/eval-tasks", method="POST")
    async def register_eval_task(
        self,
        eval_task_id: str,
        dataset_id: str,
        scoring_functions: List[str],
        provider_eval_task_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: ...
