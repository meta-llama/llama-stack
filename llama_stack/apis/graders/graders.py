# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import (
    Annotated,
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

from llama_stack.apis.datasets import DatasetPurpose
from llama_stack.apis.resource import Resource
from llama_stack.schema_utils import json_schema_type, register_schema, webmethod


class GraderType(Enum):
    """
    A type of grader. Each type is a criteria for evaluating answers.

    :cvar llm: Use an LLM to score the answer.
    :cvar regex_parser: Use a regex parser to score the answer.
    :cvar equality: Check if the answer is equal to the reference answer.
    :cvar subset_of: Check if the answer is a subset of the reference answer.
    :cvar factuality: Check if the answer is factually correct using LLM as judge.
    :cvar faithfulness: Check if the answer is faithful to the reference answer using LLM as judge.
    """

    llm = "llm"
    regex_parser = "regex_parser"
    equality = "equality"
    subset_of = "subset_of"
    factuality = "factuality"
    faithfulness = "faithfulness"


@json_schema_type
class GraderTypeInfo(BaseModel):
    """
    :param type: The type of grader.
    :param description: A description of the grader type.
        - E.g. Write your custom judge prompt to score the answer.
    :param supported_dataset_purposes: The purposes that this grader can be used for.
    """

    grader_type: GraderType
    description: str
    supported_dataset_purposes: List[DatasetPurpose] = Field(
        description="The supported purposes (supported dataset schema) that this grader can be used for. E.g. eval/question-answer",
        default_factory=list,
    )


class LlmGraderParams(BaseModel):
    model: str
    prompt: str
    score_regexes: List[str]


class RegexParserGraderParams(BaseModel):
    parsing_regexes: List[str]


@json_schema_type
class LlmGrader(BaseModel):
    type: Literal["llm"] = "llm"
    llm: LlmGraderParams


@json_schema_type
class RegexParserGrader(BaseModel):
    type: Literal["regex_parser"] = "regex_parser"
    regex_parser: RegexParserGraderParams


@json_schema_type
class EqualityGrader(BaseModel):
    type: Literal["equality"] = "equality"


@json_schema_type
class SubsetOfGrader(BaseModel):
    type: Literal["subset_of"] = "subset_of"


@json_schema_type
class FactualityGrader(BaseModel):
    type: Literal["factuality"] = "factuality"


@json_schema_type
class FaithfulnessGrader(BaseModel):
    type: Literal["faithfulness"] = "faithfulness"


GraderDefinition = register_schema(
    Annotated[
        Union[
            LlmGrader,
            RegexParserGrader,
            EqualityGrader,
            SubsetOfGrader,
            FactualityGrader,
            FaithfulnessGrader,
        ],
        Field(discriminator="type"),
    ],
    name="GraderDefinition",
)


class CommonGraderFields(BaseModel):
    grader: GraderDefinition
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata for this definition",
    )


@json_schema_type
class Grader(CommonGraderFields, Resource):
    type: Literal["grader"] = "grader"

    @property
    def grader_id(self) -> str:
        return self.identifier

    @property
    def provider_grader_id(self) -> str:
        return self.provider_resource_id


class GraderInput(CommonGraderFields, BaseModel):
    grader_id: str
    provider_id: Optional[str] = None
    provider_grader_id: Optional[str] = None


class ListGradersResponse(BaseModel):
    data: List[Grader]


class ListGraderTypesResponse(BaseModel):
    data: List[GraderTypeInfo]


@runtime_checkable
class Graders(Protocol):
    @webmethod(route="/graders", method="POST")
    async def register_grader(
        self,
        grader: GraderDefinition,
        grader_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Grader:
        """
        Register a new grader.
        :param grader: The grader definition, E.g.
            - {
                "type": "llm",
                "llm": {
                    "model": "llama-405b",
                    "prompt": "You are a judge. Score the answer based on the question. {question} {answer}",
                }
            }
        :param grader_id: (Optional) The ID of the grader. If not provided, a random ID will be generated.
        :param metadata: (Optional) Any additional metadata for this grader.
            - E.g. {
                "description": "A grader that scores the answer based on the question.",
            }
        :return: The registered grader.
        """
        ...

    @webmethod(route="/graders", method="GET")
    async def list_graders(self) -> ListGradersResponse:
        """
        List all graders.
        :return: A list of graders.
        """
        ...

    @webmethod(route="/graders/{grader_id:path}", method="GET")
    async def get_grader(self, grader_id: str) -> Grader:
        """
        Get a grader by ID.
        :param grader_id: The ID of the grader.
        :return: The grader.
        """
        ...

    @webmethod(route="/graders/{grader_id:path}", method="DELETE")
    async def unregister_grader(self, grader_id: str) -> None:
        """
        Unregister a grader by ID.
        :param grader_id: The ID of the grader.
        """
        ...

    @webmethod(route="/graders/types", method="GET")
    async def list_grader_types(self) -> ListGraderTypesResponse:
        """
        List all grader types.
        :return: A list of grader types and information about the types.
        """
        ...
