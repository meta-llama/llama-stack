# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Generic, Iterator, Literal, Protocol, TypeVar, Union

from llama_models.schema_utils import json_schema_type, webmethod
from llama_models.llama3.api.datatypes import *  # noqa: F403

from pydantic import BaseModel, Field
from typing_extensions import Annotated


@json_schema_type
class GenerationInput(BaseModel):
    messages: List[Message]


@json_schema_type
class GenerationOutput(BaseModel):
    completion_message: str
    logprobs: Optional[List[TokenLogProbs]] = None


@json_schema_type
class PostprocessedGeneration(BaseModel):
    completion_message: str
    logprobs: Optional[List[TokenLogProbs]] = None


# A sample (row) from dataset
TDatasetSample = TypeVar("TDatasetSample")


@json_schema_type
class DatasetSample(BaseModel): ...


@json_schema_type
class DictSample(DatasetSample):
    data: Dict[str, Any]


# A sample (row) from evals intermediate dataset after preprocessing
TPreprocessedSample = TypeVar("TPreprocessedSample")


@json_schema_type
class PreprocessedSample(DatasetSample):
    generation_input: GenerationInput


# A sample (row) from evals intermediate dataset after inference
TGenerationResponseSample = TypeVar("TGenerationResponseSample")


@json_schema_type
class GenerationResponseSample(DatasetSample):
    generation_output: GenerationOutput


# A sample (row) for prepared evals dataset ready for scoring
TScorerInputSample = TypeVar("TScorerInputSample")


@json_schema_type
class ScorerInputSample(DatasetSample):
    """
    A dataset is required to have the following columns to be used for scoring:
    - generated_answer: str
    - expected_answer: Union[str, List[str]]
    - (optional) input_query: str
    - (optional) generation_output: PostprocessedGeneration
    """

    generated_answer: str
    expected_answer: Union[str, List[str]]
    input_query: Optional[str] = None
    generation_output: Optional[PostprocessedGeneration] = None


@json_schema_type
class DatasetType(Enum):
    custom = "custom"
    huggingface = "huggingface"


@json_schema_type
class HuggingfaceDatasetDef(BaseModel):
    type: Literal[DatasetType.huggingface.value] = DatasetType.huggingface.value
    identifier: str = Field(
        description="A unique name for the dataset",
    )
    dataset_path: str = Field(
        description="The name of the dataset into HF (e.g. meta-llama/Llama-3.1-8B-Instruct-evals)",
    )
    dataset_name: Optional[str] = Field(
        description="The name of the dataset into HF (e.g. Llama-3.1-8B-Instruct-evals__ifeval__strict__details)",
    )
    rename_columns_map: Optional[Dict[str, str]] = Field(
        description="A map of column names to rename to fit the schema of eval dataset for scoring",
    )
    kwargs: Dict[str, Any] = Field(
        description="Any additional arguments to get Huggingface (e.g. split, trust_remote_code)",
        default_factory=dict,
    )


@json_schema_type
class CustomDatasetDef(BaseModel):
    type: Literal[DatasetType.custom.value] = DatasetType.custom.value
    identifier: str = Field(
        description="A unique name for the dataset",
    )
    url: str = Field(
        description="The URL to the dataset",
    )


DatasetDef = Annotated[
    Union[
        HuggingfaceDatasetDef,
        CustomDatasetDef,
    ],
    Field(discriminator="type"),
]


class DatasetsResponseStatus(Enum):
    success = "success"
    fail = "fail"


@json_schema_type
class CreateDatasetResponse(BaseModel):
    status: DatasetsResponseStatus = Field(
        description="Return status of the dataset creation",
    )
    msg: Optional[str] = None


@json_schema_type
class DeleteDatasetResponse(BaseModel):
    status: DatasetsResponseStatus = Field(
        description="Return status of the dataset creation",
    )
    msg: Optional[str] = None


class BaseDataset(ABC, Generic[TDatasetSample]):
    def __init__(self) -> None:
        self.type: str = self.__class__.__name__

    @property
    @abstractmethod
    def dataset_id(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self) -> Iterator[TDatasetSample]:
        raise NotImplementedError()

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError()


class Datasets(Protocol):
    @webmethod(route="/datasets/create")
    async def create_dataset(
        self,
        dataset_def: DatasetDef,
    ) -> CreateDatasetResponse: ...

    @webmethod(route="/datasets/get", method="GET")
    async def get_dataset(
        self,
        dataset_identifier: str,
    ) -> Optional[DatasetDef]: ...

    @webmethod(route="/datasets/delete")
    async def delete_dataset(
        self,
        dataset_identifier: str,
    ) -> DeleteDatasetResponse: ...

    @webmethod(route="/datasets/list", method="GET")
    async def list_datasets(self) -> List[DatasetDef]: ...
