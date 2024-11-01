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


@json_schema_type
class DatasetDef(BaseModel):
    identifier: str = Field(
        description="A unique name for the dataset",
    )
    dataset_schema: Dict[str, ParamType] = Field(
        description="The schema definition for this dataset",
    )
    url: URL
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata for this dataset",
    )


@json_schema_type
class DatasetDefWithProvider(DatasetDef):
    type: Literal["dataset"] = "dataset"
    provider_id: str = Field(
        description="ID of the provider which serves this dataset",
    )


class Datasets(Protocol):
    @webmethod(route="/datasets/register", method="POST")
    async def register_dataset(
        self,
        dataset_def: DatasetDefWithProvider,
    ) -> None: ...

    @webmethod(route="/datasets/get", method="GET")
    async def get_dataset(
        self,
        dataset_identifier: str,
    ) -> Optional[DatasetDefWithProvider]: ...

    @webmethod(route="/datasets/list", method="GET")
    async def list_datasets(self) -> List[DatasetDefWithProvider]: ...
