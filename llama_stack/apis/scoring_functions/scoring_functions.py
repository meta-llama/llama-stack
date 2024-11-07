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


# Perhaps more structure can be imposed on these functions. Maybe they could be associated
# with standard metrics so they can be rolled up?
@json_schema_type
class ScoringConfigType(Enum):
    llm_as_judge = "llm_as_judge"
    regex_parser = "regex_parser"


@json_schema_type
class LLMAsJudgeScoringFnParams(BaseModel):
    type: Literal[ScoringConfigType.llm_as_judge.value] = (
        ScoringConfigType.llm_as_judge.value
    )
    judge_model: str
    prompt_template: Optional[str] = None
    judge_score_regexes: Optional[List[str]] = Field(
        description="Regexes to extract the answer from generated response",
        default_factory=list,
    )


@json_schema_type
class RegexParserScoringFnParams(BaseModel):
    type: Literal[ScoringConfigType.regex_parser.value] = (
        ScoringConfigType.regex_parser.value
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


@json_schema_type
class ScoringFnDef(BaseModel):
    identifier: str
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
    # We can optionally add information here to support packaging of code, etc.


@json_schema_type
class ScoringFnDefWithProvider(ScoringFnDef):
    type: Literal["scoring_fn"] = "scoring_fn"
    provider_id: str = Field(
        description="ID of the provider which serves this dataset",
    )


@runtime_checkable
class ScoringFunctions(Protocol):
    @webmethod(route="/scoring_functions/list", method="GET")
    async def list_scoring_functions(self) -> List[ScoringFnDefWithProvider]: ...

    @webmethod(route="/scoring_functions/get", method="GET")
    async def get_scoring_function(
        self, name: str
    ) -> Optional[ScoringFnDefWithProvider]: ...

    @webmethod(route="/scoring_functions/register", method="POST")
    async def register_scoring_function(
        self, function_def: ScoringFnDefWithProvider
    ) -> None: ...
