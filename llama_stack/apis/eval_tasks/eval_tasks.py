# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

from llama_models.schema_utils import json_schema_type, webmethod

from pydantic import Field

from llama_stack.apis.resource import Resource


@json_schema_type
class EvalTask(Resource):
    type: Literal["eval_task"] = "eval_task"
    dataset_id: str
    scoring_functions: List[str]
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata for this evaluation task",
    )


@runtime_checkable
class EvalTasks(Protocol):
    @webmethod(route="/eval_tasks/list", method="GET")
    async def list_eval_tasks(self) -> List[EvalTask]: ...

    @webmethod(route="/eval_tasks/get", method="GET")
    async def get_eval_task(self, name: str) -> Optional[EvalTask]: ...

    @webmethod(route="/eval_tasks/register", method="POST")
    async def register_eval_task(
        self,
        eval_task_id: str,
        dataset_id: str,
        scoring_functions: List[str],
        provider_id: Optional[str] = None,
        provider_eval_task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: ...
