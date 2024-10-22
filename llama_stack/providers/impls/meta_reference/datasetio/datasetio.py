# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import List, Optional

from llama_models.llama3.api.datatypes import *  # noqa: F403

from llama_stack.apis.datasetio import *  # noqa: F403
from llama_stack.providers.datatypes import DatasetsProtocolPrivate

from .config import MetaReferenceDatasetIOConfig


class MetaReferenceDatasetioImpl(DatasetIO, DatasetsProtocolPrivate):
    def __init__(self, config: MetaReferenceDatasetIOConfig) -> None:
        self.config = config

    async def initialize(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def register_dataset(
        self,
        memory_bank: DatasetDef,
    ) -> None:
        print("register dataset")

    async def list_datasets(self) -> List[DatasetDef]:
        print("list datasets")
        return []

    async def get_rows_paginated(
        self,
        dataset_id: str,
        rows_in_page: int,
        page_token: Optional[str] = None,
        filter_condition: Optional[str] = None,
    ) -> PaginatedRowsResult:
        print("get rows paginated")

        return PaginatedRowsResult(rows=[], total_count=1, next_page_token=None)
