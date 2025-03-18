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
class IterrowsResponse(BaseModel):
    """
    A paginated list of rows from a dataset.

    :param data: The rows in the current page.
    :param next_start_index: Index into dataset for the first row in the next page. None if there are no more rows.
    """

    data: List[Dict[str, Any]]
    next_start_index: Optional[int] = None


class DatasetStore(Protocol):
    def get_dataset(self, dataset_id: str) -> Dataset: ...


@runtime_checkable
class DatasetIO(Protocol):
    # keeping for aligning with inference/safety, but this is not used
    dataset_store: DatasetStore

    # TODO(xiyan): there's a flakiness here where setting route to "/datasets/" here will not result in proper routing
    @webmethod(route="/datasetio/iterrows/{dataset_id:path}", method="GET")
    async def iterrows(
        self,
        dataset_id: str,
        start_index: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> IterrowsResponse:
        """Get a paginated list of rows from a dataset. Uses cursor-based pagination.

        :param dataset_id: The ID of the dataset to get the rows from.
        :param start_index: Index into dataset for the first row to get. Get all rows if None.
        :param limit: The number of rows to get.
        """
        ...

    @webmethod(route="/datasetio/append-rows/{dataset_id:path}", method="POST")
    async def append_rows(self, dataset_id: str, rows: List[Dict[str, Any]]) -> None: ...
