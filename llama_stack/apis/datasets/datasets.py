# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Protocol, Annotated, Union

from pydantic import BaseModel, Field

from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.schema_utils import json_schema_type, webmethod, register_schema


class Schema(Enum):
    """
    Schema of the dataset. Each type has a different column format.
    :cvar jsonl_messages: The dataset is a JSONL file with messages. Examples:
        {
            "messages": [
                {"role": "user", "content": "Hello, world!"},
                {"role": "assistant", "content": "Hello, world!"},
            ]
        }
    """

    jsonl_messages = "jsonl_messages"
    # TODO: add more schemas here


class DatasetType(Enum):
    huggingface = "huggingface"
    uri = "uri"
    rows = "rows"


@json_schema_type
class URIDataReference(BaseModel):
    type: Literal["uri"] = "uri"
    uri: str


@json_schema_type
class HuggingfaceDataReference(BaseModel):
    type: Literal["huggingface"] = "huggingface"
    dataset_path: str
    params: Dict[str, Any]


@json_schema_type
class RowsDataReference(BaseModel):
    type: Literal["rows"] = "rows"
    rows: List[Dict[str, Any]]


DataReference = register_schema(
    Annotated[
        Union[URIDataReference, HuggingfaceDataReference, RowsDataReference],
        Field(discriminator="type"),
    ],
    name="DataReference",
)

class CommonDatasetFields(BaseModel):
    schema: Schema
    data_reference: DataReference
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
        data_reference: DataReference,
        metadata: Optional[Dict[str, Any]] = None,
        dataset_id: Optional[str] = None,
    ) -> Dataset:
        """
        Register a new dataset through a file or

        :param schema: The schema format of the dataset. One of
            - jsonl_messages: The dataset is a JSONL file with messages in column format
        :param data_reference: The data reference of the dataset. Examples:
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
               "rows": [{"message": "Hello, world!"}]
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
