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
    Union,
    runtime_checkable,
)

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from llama_stack.apis.common.type_system import ParamType
from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.schema_utils import json_schema_type, register_schema, webmethod


# Perhaps more structure can be imposed on these functions. Maybe they could be associated
# with standard metrics so they can be rolled up?
@json_schema_type
class ScoringFnParamsType(Enum):
    llm_as_judge = "llm_as_judge"
    regex_parser = "regex_parser"
    basic = "basic"


@json_schema_type
class AggregationFunctionType(Enum):
    average = "average"
    weighted_average = "weighted_average"
    median = "median"
    categorical_count = "categorical_count"
    accuracy = "accuracy"


@json_schema_type
class LLMAsJudgeScoringFnParams(BaseModel):
    type: Literal[ScoringFnParamsType.llm_as_judge.value] = ScoringFnParamsType.llm_as_judge.value
    judge_model: str
    prompt_template: Optional[str] = None
    judge_score_regexes: Optional[List[str]] = Field(
        description="Regexes to extract the answer from generated response",
        default_factory=list,
    )
    aggregation_functions: Optional[List[AggregationFunctionType]] = Field(
        description="Aggregation functions to apply to the scores of each row",
        default_factory=list,
    )


@json_schema_type
class RegexParserScoringFnParams(BaseModel):
    type: Literal[ScoringFnParamsType.regex_parser.value] = ScoringFnParamsType.regex_parser.value
    parsing_regexes: Optional[List[str]] = Field(
        description="Regex to extract the answer from generated response",
        default_factory=list,
    )
    aggregation_functions: Optional[List[AggregationFunctionType]] = Field(
        description="Aggregation functions to apply to the scores of each row",
        default_factory=list,
    )


@json_schema_type
class BasicScoringFnParams(BaseModel):
    type: Literal[ScoringFnParamsType.basic.value] = ScoringFnParamsType.basic.value
    aggregation_functions: Optional[List[AggregationFunctionType]] = Field(
        description="Aggregation functions to apply to the scores of each row",
        default_factory=list,
    )


ScoringFnParams = Annotated[
    Union[
        LLMAsJudgeScoringFnParams,
        RegexParserScoringFnParams,
        BasicScoringFnParams,
    ],
    Field(discriminator="type"),
]
register_schema(ScoringFnParams, name="ScoringFnParams")


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
    type: Literal[ResourceType.scoring_function.value] = ResourceType.scoring_function.value

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


class ListScoringFunctionsResponse(BaseModel):
    data: List[ScoringFn]


@runtime_checkable
class ScoringFunctions(Protocol):
    @webmethod(route="/scoring-functions", method="GET")
    async def list_scoring_functions(self) -> ListScoringFunctionsResponse: ...

    @webmethod(route="/scoring-functions/{scoring_fn_id:path}", method="GET")
    async def get_scoring_function(self, scoring_fn_id: str, /) -> ScoringFn: ...

    @webmethod(route="/scoring-functions", method="POST")
    async def register_scoring_function(
        self,
        scoring_fn_id: str,
        description: str,
        return_type: ParamType,
        provider_scoring_fn_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        params: Optional[ScoringFnParams] = None,
    ) -> None: ...
