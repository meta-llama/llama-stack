# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    runtime_checkable,
    Union,
)

from llama_models.schema_utils import json_schema_type, webmethod
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from llama_stack.apis.common.type_system import ParamType


@json_schema_type
class Parameter(BaseModel):
    name: str
    type: ParamType
    description: Optional[str] = None


# Perhaps more structure can be imposed on these functions. Maybe they could be associated
# with standard metrics so they can be rolled up?


@json_schema_type
class CommonDef(BaseModel):
    name: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata for this definition",
    )
    # Hack: same with memory_banks for union defs
    provider_id: str = ""


@json_schema_type
class DeterministicFunctionDef(CommonDef):
    type: Literal["deterministic"] = "deterministic"
    parameters: List[Parameter] = Field(
        description="List of parameters for the deterministic function",
    )
    return_type: ParamType = Field(
        description="The return type of the deterministic function",
    )
    # We can optionally add information here to support packaging of code, etc.


@json_schema_type
class LLMJudgeFunctionDef(CommonDef):
    type: Literal["judge"] = "judge"
    model: str = Field(
        description="The LLM model to use for the judge function",
    )


ScoringFunctionDef = Annotated[
    Union[DeterministicFunctionDef, LLMJudgeFunctionDef], Field(discriminator="type")
]

ScoringFunctionDefWithProvider = ScoringFunctionDef


@runtime_checkable
class ScoringFunctions(Protocol):
    @webmethod(route="/scoring_functions/list", method="GET")
    async def list_scoring_functions(self) -> List[ScoringFunctionDefWithProvider]: ...

    @webmethod(route="/scoring_functions/get", method="GET")
    async def get_scoring_function(
        self, name: str
    ) -> Optional[ScoringFunctionDefWithProvider]: ...

    @webmethod(route="/scoring_functions/register", method="POST")
    async def register_scoring_function(
        self, function: ScoringFunctionDefWithProvider
    ) -> None: ...
