# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel

from llama_stack.apis.datasets import Dataset
from llama_stack.schema_utils import json_schema_type, webmethod


@json_schema_type
class PaginatedRowsResult(BaseModel):
    """
    A paginated list of rows from a dataset.

    :param rows: The rows in the current page.
    :param total_count: The total number of rows in the dataset.
    :param next_page_token: The token to get the next page of rows.
    """

    # the rows obey the DatasetSchema for the given dataset
    rows: List[Dict[str, Any]]
    total_count: int
    next_page_token: Optional[str] = None


class DatasetStore(Protocol):
    def get_dataset(self, dataset_id: str) -> Dataset: ...


@runtime_checkable
class DatasetIO(Protocol):
    # keeping for aligning with inference/safety, but this is not used
    dataset_store: DatasetStore

    @webmethod(route="/datasetio/rows", method="GET")
    async def get_rows_paginated(
        self,
        dataset_id: str,
        rows_in_page: int,
        page_token: Optional[str] = None,
        filter_condition: Optional[str] = None,
    ) -> PaginatedRowsResult:
        """Get a paginated list of rows from a dataset.

        :param dataset_id: The ID of the dataset to get the rows from.
        :param rows_in_page: The number of rows to get per page.
        :param page_token: The token to get the next page of rows.
        :param filter_condition: (Optional) A condition to filter the rows by.
        """
        ...

    @webmethod(route="/datasetio/rows", method="POST")
    async def append_rows(self, dataset_id: str, rows: List[Dict[str, Any]]) -> None: ...
