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

# A sample (row) from dataset
TDatasetSample = TypeVar("TDatasetSample")


@json_schema_type
class DatasetSample(BaseModel): ...


@json_schema_type
class DictSample(DatasetSample):
    data: Dict[str, Any]


# A sample (row) from evals intermediate dataset
TProcessedSample = TypeVar("TProcessedSample")


@json_schema_type
class PredictionSample(BaseModel):
    completion_message: str


@json_schema_type
class ProcessedDictSample(DictSample):
    preprocessed: Optional[Dict[str, Any]] = None
    prediction: Optional[PredictionSample] = None
    postprocessed: Optional[Dict[str, Any]] = None


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
    dataset_name: str = Field(
        description="The name of the dataset into HF (e.g. hellawag)",
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
    def create_dataset(
        self,
        dataset: DatasetDef,
    ) -> None: ...

    @webmethod(route="/datasets/get")
    def get_dataset(
        self,
        dataset_identifier: str,
    ) -> DatasetDef: ...

    @webmethod(route="/datasets/delete")
    def delete_dataset(
        self,
        dataset_uuid: str,
    ) -> None: ...
