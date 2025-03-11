# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Protocol

from pydantic import BaseModel, Field

from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.schema_utils import json_schema_type, webmethod


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


class CommonDatasetFields(BaseModel):
    schema: Schema
    uri: str
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
        uri: str,
        uri_params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        dataset_id: Optional[str] = None,
    ) -> Dataset:
        """
        Register a new dataset through a file or

        :param schema: The schema format of the dataset. One of
            - jsonl_messages: The dataset is a JSONL file with messages in column format
        :param uri: The URI of the dataset. Examples:
           - file://mydata.jsonl
           - s3://mybucket/myfile.jsonl
           - https://mywebsite.com/myfile.jsonl
           - huggingface://tatsu-lab/alpaca
        :param uri_params: The parameters for the URI.
           - E.g. If URL is a huggingface dataset, parameters could be uri_params={"split": "train"}
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
