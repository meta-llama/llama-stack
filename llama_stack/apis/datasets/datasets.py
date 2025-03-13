# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Protocol, Union

from pydantic import BaseModel, Field

from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.schema_utils import json_schema_type, register_schema, webmethod


class DatasetPurpose(Enum):
    """
    Purpose of the dataset. Each type has a different column format.
    :cvar post-training/messages: The dataset contains messages used for post-training. Examples:
        {
            "messages": [
                {"role": "user", "content": "Hello, world!"},
                {"role": "assistant", "content": "Hello, world!"},
            ]
        }
    """

    post_training_messages = "post-training/messages"
    eval_question_answer = "eval/question-answer"

    # TODO: add more schemas here


class DatasetType(Enum):
    """
    Type of the dataset source.
    :cvar huggingface: The dataset is stored in Huggingface.
    :cvar uri: The dataset can be obtained from a URI. 
    :cvar rows: The dataset is stored in rows. 
    """
    huggingface = "huggingface"
    uri = "uri"
    rows = "rows"


@json_schema_type
class URIDataSource(BaseModel):
    """A dataset that can be obtained from a URI.
    :param uri: The dataset can be obtained from a URI. E.g.
        - "https://mywebsite.com/mydata.jsonl"
        - "lsfs://mydata.jsonl"
        - "data:csv;base64,{base64_content}"
    """
    type: Literal["uri"] = "uri"
    uri: str


@json_schema_type
class HuggingfaceDataSource(BaseModel):
    """A dataset stored in Huggingface.
    :param path: The path to the dataset in Huggingface. E.g.
        - "llamastack/simpleqa"
    :param params: The parameters for the dataset.
    """
    type: Literal["huggingface"] = "huggingface"
    path: str
    params: Dict[str, Any]


@json_schema_type
class RowsDataSource(BaseModel):
    """A dataset stored in rows.
    :param rows: The dataset is stored in rows. E.g.
        - [
            {"messages": [{"role": "user", "content": "Hello, world!"}, {"role": "assistant", "content": "Hello, world!"}]}
        ]
    """
    type: Literal["rows"] = "rows"
    rows: List[Dict[str, Any]]


DataSource = register_schema(
    Annotated[
        Union[URIDataSource, HuggingfaceDataSource, RowsDataSource],
        Field(discriminator="type"),
    ],
    name="DataSource",
)


class CommonDatasetFields(BaseModel):
    """
    Common fields for a dataset.
    """
    purpose: DatasetPurpose
    source: DataSource
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata for this dataset",
    )


@json_schema_type
class Dataset(CommonDatasetFields, Resource):
    type: Literal[ResourceType.dataset.value] = ResourceType.dataset.value

    @property
    def dataset_id(self) -> str:
        return self.identifier

    @property
    def provider_dataset_id(self) -> str:
        return self.provider_resource_id


class DatasetInput(CommonDatasetFields, BaseModel):
    dataset_id: str
    provider_id: Optional[str] = None
    provider_dataset_id: Optional[str] = None


class ListDatasetsResponse(BaseModel):
    data: List[Dataset]


class Datasets(Protocol):
    @webmethod(route="/datasets", method="POST")
    async def register_dataset(
        self,
        purpose: DatasetPurpose,
        source: DataSource,
        metadata: Optional[Dict[str, Any]] = None,
        dataset_id: Optional[str] = None,
    ) -> Dataset:
        """
        Register a new dataset.

        :param purpose: The purpose of the dataset. One of
            - "post-training/messages": The dataset contains a messages column with list of messages for post-training.
            - "eval/question-answer": The dataset contains a question and answer column.
        :param source: The data source of the dataset. Examples:
           - {
               "type": "uri",
               "uri": "https://mywebsite.com/mydata.jsonl"
           }
           - {
               "type": "uri",
               "uri": "lsfs://mydata.jsonl"
           }
           - {
               "type": "huggingface",
               "dataset_path": "tatsu-lab/alpaca",
               "params": {
                   "split": "train"
               }
           }
           - {
               "type": "rows",
               "rows": [
                   {
                       "messages": [
                           {"role": "user", "content": "Hello, world!"},
                           {"role": "assistant", "content": "Hello, world!"},
                       ]
                   }
               ]
           }
        :param metadata: The metadata for the dataset.
           - E.g. {"description": "My dataset"}
        :param dataset_id: The ID of the dataset. If not provided, a random ID will be generated.
        """
        ...

    @webmethod(route="/datasets/{dataset_id:path}", method="GET")
    async def get_dataset(
        self,
        dataset_id: str,
    ) -> Optional[Dataset]: ...

    @webmethod(route="/datasets", method="GET")
    async def list_datasets(self) -> ListDatasetsResponse: ...

    @webmethod(route="/datasets/{dataset_id:path}", method="DELETE")
    async def unregister_dataset(
        self,
        dataset_id: str,
    ) -> None: ...
