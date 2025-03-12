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


class Schema(Enum):
    """
    Schema of the dataset. Each type has a different column format.
    :cvar messages: The dataset contains messages used for post-training. Examples:
        {
            "messages": [
                {"role": "user", "content": "Hello, world!"},
                {"role": "assistant", "content": "Hello, world!"},
            ]
        }
    """

    messages = "messages"
    # TODO: add more schemas here


class DatasetType(Enum):
    huggingface = "huggingface"
    uri = "uri"
    rows = "rows"


@json_schema_type
class URIDataSource(BaseModel):
    type: Literal["uri"] = "uri"
    uri: str


@json_schema_type
class HuggingfaceDataSource(BaseModel):
    type: Literal["huggingface"] = "huggingface"
    dataset_path: str
    params: Dict[str, Any]


@json_schema_type
class RowsDataSource(BaseModel):
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
    schema: Schema
    data_source: DataSource
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
        schema: Schema,
        data_source: DataSource,
        metadata: Optional[Dict[str, Any]] = None,
        dataset_id: Optional[str] = None,
    ) -> Dataset:
        """
        Register a new dataset.

        :param schema: The schema format of the dataset. One of
            - messages: The dataset contains a messages column with list of messages for post-training.
        :param data_source: The data source of the dataset. Examples:
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
