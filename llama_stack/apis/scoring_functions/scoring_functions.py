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

from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.schema_utils import json_schema_type, register_schema, webmethod
from llama_stack.apis.datasets import DatasetPurpose

# Perhaps more structure can be imposed on these functions. Maybe they could be associated
# with standard metrics so they can be rolled up?


class ScoringFunctionType(Enum):
    """
    A type of scoring function. Each type is a criteria for evaluating answers.

    :cvar llm_as_judge: Scoring function that uses a judge model to score the answer.
    :cvar regex_parser: Scoring function that parses the answer from the generated response using regexes, and checks against the expected answer.
    """

    custom_llm_as_judge = "custom_llm_as_judge"
    regex_parser = "regex_parser"
    regex_parser_math_response = "regex_parser_math_response"
    equality = "equality"
    subset_of = "subset_of"
    factuality = "factuality"
    faithfulness = "faithfulness"
    answer_correctness = "answer_correctness"
    answer_relevancy = "answer_relevancy"
    answer_similarity = "answer_similarity"
    context_entity_recall = "context_entity_recall"
    context_precision = "context_precision"
    context_recall = "context_recall"
    context_relevancy = "context_relevancy"


class AggregationFunctionType(Enum):
    """
    A type of aggregation function.

    :cvar average: Average the scores of each row.
    :cvar median: Median the scores of each row.
    :cvar categorical_count: Count the number of rows that match each category.
    :cvar accuracy: Number of correct results over total results.
    """

    average = "average"
    median = "median"
    categorical_count = "categorical_count"
    accuracy = "accuracy"


class BasicScoringFnParams(BaseModel):
    """
    :param aggregation_functions: (Optional) Aggregation functions to apply to the scores of each row. If not provided, no aggregation will be performed.
    """

    aggregation_functions: Optional[List[AggregationFunctionType]] = Field(
        description="Aggregation functions to apply to the scores of each row",
        default_factory=list,
    )


class RegexParserScoringFnParams(BaseModel):
    """
    :param parsing_regexes: (Optional) Regexes to extract the answer from generated response.
    :param aggregation_functions: (Optional) Aggregation functions to apply to the scores of each row. If not provided, no aggregation will be performed.
    """

    parsing_regexes: List[str] = Field(
        description="Regexes to extract the answer from generated response",
        default_factory=list,
    )
    aggregation_functions: Optional[List[AggregationFunctionType]] = Field(
        description="Aggregation functions to apply to the scores of each row",
        default_factory=list,
    )

class CustomLLMAsJudgeScoringFnParams(BaseModel):
    type: Literal["custom_llm_as_judge"] = "custom_llm_as_judge"
    judge_model: str
    prompt_template: Optional[str] = None
    judge_score_regexes: Optional[List[str]] = Field(
        description="Regexes to extract the answer from generated response",
        default_factory=list,
    )

@json_schema_type
class RegexParserScoringFn(BaseModel):
    type: Literal["regex_parser"] = "regex_parser"
    regex_parser: RegexParserScoringFnParams


@json_schema_type
class RegexParserMathScoringFn(BaseModel):
    type: Literal["regex_parser_math_response"] = "regex_parser_math_response"
    regex_parser_math_response: RegexParserScoringFnParams

@json_schema_type
class EqualityScoringFn(BaseModel):
    type: Literal["equality"] = "equality"
    equality: BasicScoringFnParams

@json_schema_type
class SubsetOfScoringFn(BaseModel):
    type: Literal["subset_of"] = "subset_of"
    subset_of: BasicScoringFnParams

@json_schema_type
class FactualityScoringFn(BaseModel):
    type: Literal["factuality"] = "factuality"
    factuality: BasicScoringFnParams

@json_schema_type
class FaithfulnessScoringFn(BaseModel):
    type: Literal["faithfulness"] = "faithfulness"
    faithfulness: BasicScoringFnParams

@json_schema_type
class AnswerCorrectnessScoringFn(BaseModel):
    type: Literal["answer_correctness"] = "answer_correctness"
    answer_correctness: BasicScoringFnParams

@json_schema_type
class AnswerRelevancyScoringFn(BaseModel):
    type: Literal["answer_relevancy"] = "answer_relevancy"
    answer_relevancy: BasicScoringFnParams

@json_schema_type
class AnswerSimilarityScoringFn(BaseModel):
    type: Literal["answer_similarity"] = "answer_similarity"
    answer_similarity: BasicScoringFnParams


@json_schema_type
class ContextEntityRecallScoringFn(BaseModel):
    type: Literal["context_entity_recall"] = "context_entity_recall"
    context_entity_recall: BasicScoringFnParams


@json_schema_type
class ContextPrecisionScoringFn(BaseModel):
    type: Literal["context_precision"] = "context_precision"
    context_precision: BasicScoringFnParams


@json_schema_type
class ContextRecallScoringFn(BaseModel):
    type: Literal["context_recall"] = "context_recall"
    context_recall: BasicScoringFnParams


@json_schema_type
class ContextRelevancyScoringFn(BaseModel):
    type: Literal["context_relevancy"] = "context_relevancy"
    context_relevancy: BasicScoringFnParams


@json_schema_type
class CustomLLMAsJudgeScoringFn(BaseModel):
    type: Literal["custom_llm_as_judge"] = "custom_llm_as_judge"
    custom_llm_as_judge: CustomLLMAsJudgeScoringFnParams


ScoringFnDefinition = register_schema(
    Annotated[
        Union[
            CustomLLMAsJudgeScoringFn,
            RegexParserScoringFn,
            RegexParserMathScoringFn,
            EqualityScoringFn,
            SubsetOfScoringFn,
            FactualityScoringFn,
            FaithfulnessScoringFn,
            AnswerCorrectnessScoringFn,
            AnswerRelevancyScoringFn,
            AnswerSimilarityScoringFn,
            ContextEntityRecallScoringFn,
            ContextPrecisionScoringFn,
            ContextRecallScoringFn,
            ContextRelevancyScoringFn,
        ],
        Field(discriminator="type"),
    ],
    name="ScoringFnDefinition",
)


class CommonScoringFnFields(BaseModel):
    """
    :param fn: The scoring function type and parameters. 
    :param metadata: (Optional) Any additional metadata for this definition (e.g. description).
    """
    fn: ScoringFnDefinition
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata for this definition (e.g. description)",
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


@json_schema_type
class ScoringFnTypeInfo(BaseModel):
    """
    :param type: The type of scoring function. 
    :param description: A description of the scoring function type. 
        - E.g. Write your custom judge prompt to score the answer. 
    :param supported_purposes: The purposes that this scoring function can be used for.
    """
    type: ScoringFunctionType
    description: str
    supported_purposes: List[DatasetPurpose] = Field(
        description="The supported purposes (supported dataset schema) that this scoring function can be used for. E.g. eval/question-answer",
        default_factory=list,
    )


class ScoringFnInput(CommonScoringFnFields, BaseModel):
    scoring_fn_id: str
    provider_id: Optional[str] = None
    provider_scoring_fn_id: Optional[str] = None


class ListScoringFunctionsResponse(BaseModel):
    data: List[ScoringFn]


class ListScoringFunctionTypesResponse(BaseModel):
    data: List[ScoringFnTypeInfo]


@runtime_checkable
class ScoringFunctions(Protocol):
    @webmethod(route="/scoring-functions", method="GET")
    async def list_scoring_functions(self) -> ListScoringFunctionsResponse: 
        """
        List all registered scoring functions.
        """
        ...

    @webmethod(route="/scoring-functions/types", method="GET")
    async def list_scoring_function_types(self) -> ListScoringFunctionTypesResponse: 
        """
        List all available scoring function types information and how to use them. 
        """
        ...

    @webmethod(route="/scoring-functions/{scoring_fn_id:path}", method="GET")
    async def get_scoring_function(
        self,
        scoring_fn_id: str,
    ) -> Optional[ScoringFn]: 
        """
        Get a scoring function by its ID.
        :param scoring_fn_id: The ID of the scoring function to get.
        """
        ...

    @webmethod(route="/scoring-functions", method="POST")
    async def register_scoring_function(
        self,
        fn: ScoringFnDefinition,
        scoring_fn_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ScoringFn:
        """
        Register a new scoring function with given parameters.
        Only valid scoring function type that can be parameterized can be registered.

        :param fn: The type and parameters for the scoring function.
        :param scoring_fn_id: (Optional) The ID of the scoring function to register. If not provided, a random ID will be generated.
        :param metadata: (Optional) Any additional metadata to be associated with the scoring function.
            - E.g. {"description": "This scoring function is used for ..."}
        """
        ...
    
    @webmethod(route="/scoring-functions/{scoring_fn_id:path}", method="DELETE")
    async def unregister_scoring_function(
        self,
        scoring_fn_id: str,
    ) -> None: 
        """
        Unregister a scoring function by its ID.
        :param scoring_fn_id: The ID of the scoring function to unregister.
        """
        ...
