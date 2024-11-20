# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
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

from llama_stack.apis.resource import Resource, ResourceType


# Perhaps more structure can be imposed on these functions. Maybe they could be associated
# with standard metrics so they can be rolled up?
@json_schema_type
class ScoringFnParamsType(Enum):
    llm_as_judge = "llm_as_judge"
    regex_parser = "regex_parser"


@json_schema_type
class LLMAsJudgeScoringFnParams(BaseModel):
    type: Literal[ScoringFnParamsType.llm_as_judge.value] = (
        ScoringFnParamsType.llm_as_judge.value
    )
    judge_model: str
    prompt_template: Optional[str] = None
    judge_score_regexes: Optional[List[str]] = Field(
        description="Regexes to extract the answer from generated response",
        default_factory=list,
    )


@json_schema_type
class RegexParserScoringFnParams(BaseModel):
    type: Literal[ScoringFnParamsType.regex_parser.value] = (
        ScoringFnParamsType.regex_parser.value
    )
    parsing_regexes: Optional[List[str]] = Field(
        description="Regex to extract the answer from generated response",
        default_factory=list,
    )


ScoringFnParams = Annotated[
    Union[
        LLMAsJudgeScoringFnParams,
        RegexParserScoringFnParams,
    ],
    Field(discriminator="type"),
]


class CommonScoringFnFields(BaseModel):
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata for this definition",
    )
    return_type: ParamType = Field(
        description="The return type of the deterministic function",
    )
    params: Optional[ScoringFnParams] = Field(
        description="The parameters for the scoring function for benchmark eval, these can be overridden for app eval",
        default=None,
    )


@json_schema_type
class ScoringFn(CommonScoringFnFields, Resource):
    type: Literal[ResourceType.scoring_function.value] = (
        ResourceType.scoring_function.value
    )

    @property
    def scoring_fn_id(self) -> str:
        return self.identifier

    @property
    def provider_scoring_fn_id(self) -> str:
        return self.provider_resource_id


class ScoringFnInput(CommonScoringFnFields, BaseModel):
    scoring_fn_id: str
    provider_id: Optional[str] = None
    provider_scoring_fn_id: Optional[str] = None


@runtime_checkable
class ScoringFunctions(Protocol):
    @webmethod(route="/scoring-functions/list", method="GET")
    async def list_scoring_functions(self) -> List[ScoringFn]: ...

    @webmethod(route="/scoring-functions/get", method="GET")
    async def get_scoring_function(self, scoring_fn_id: str) -> Optional[ScoringFn]: ...

    @webmethod(route="/scoring-functions/register", method="POST")
    async def register_scoring_function(
        self,
        scoring_fn_id: str,
        description: str,
        return_type: ParamType,
        provider_scoring_fn_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        params: Optional[ScoringFnParams] = None,
    ) -> None: ...
