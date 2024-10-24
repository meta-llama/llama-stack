# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from llama_models.schema_utils import json_schema_type, webmethod
from pydantic import BaseModel

from llama_stack.apis.datasets import *  # noqa: F403


@json_schema_type
class PaginatedRowsResult(BaseModel):
    # the rows obey the DatasetSchema for the given dataset
    rows: List[Dict[str, Any]]
    total_count: int
    next_page_token: Optional[str] = None


class DatasetStore(Protocol):
    def get_dataset(self, identifier: str) -> DatasetDefWithProvider: ...


@runtime_checkable
class DatasetIO(Protocol):
    # keeping for aligning with inference/safety, but this is not used
    dataset_store: DatasetStore

    @webmethod(route="/datasetio/get_rows_paginated", method="GET")
    async def get_rows_paginated(
        self,
        dataset_id: str,
        rows_in_page: int,
        page_token: Optional[str] = None,
        filter_condition: Optional[str] = None,
    ) -> PaginatedRowsResult: ...
