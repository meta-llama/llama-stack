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
    answer_parsing = "answer_parsing"


@json_schema_type
class LLMAsJudgeScoringFnConfig(BaseModel):
    type: Literal[ScoringContextType.llm_as_judge.value] = (  # type: ignore
        ScoringConfigType.llm_as_judge.value
    )
    judge_model: str
    prompt_template: Optional[str] = None
    judge_score_regex: Optional[List[str]] = Field()


@json_schema_type
class AnswerParsingScoringFnConfig(BaseModel):
    type: Literal[ScoringContextType.answer_parsing.value] = (  # type: ignore
        ScoringConfigType.answer_parsing.value
    )
    parsing_regex: Optional[List[str]] = Field(
        description="Regex to extract the answer from generated response",
        default_factory=list,
    )


ScoringFnConfig = Annotated[
    Union[
        LLMAsJudgeScoringFnConfig,
        AnswerParsingScoringFnConfig,
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
    config: Optional[ScoringFnConfig] = None  # type: ignore
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
