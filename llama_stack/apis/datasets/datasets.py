# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Literal, Optional, Protocol

from llama_models.llama3.api.datatypes import URL

from llama_models.schema_utils import json_schema_type, webmethod

from pydantic import BaseModel, Field

from llama_stack.apis.common.type_system import ParamType
from llama_stack.apis.resource import Resource, ResourceType


class CommonDatasetFields(BaseModel):
    dataset_schema: Dict[str, ParamType]
    url: URL
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


class Datasets(Protocol):
    @webmethod(route="/datasets/register", method="POST")
    async def register_dataset(
        self,
        dataset_id: str,
        dataset_schema: Dict[str, ParamType],
        url: URL,
        provider_dataset_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: ...

    @webmethod(route="/datasets/get", method="GET")
    async def get_dataset(
        self,
        dataset_id: str,
    ) -> Optional[Dataset]: ...

    @webmethod(route="/datasets/list", method="GET")
    async def list_datasets(self) -> List[Dataset]: ...

    @webmethod(route="/datasets/unregister", method="POST")
    async def unregister_dataset(
        self,
        dataset_id: str,
    ) -> None: ...
