# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any, Dict, List, Optional

import pandas
from llama_models.llama3.api.datatypes import *  # noqa: F403

from llama_stack.apis.datasetio import *  # noqa: F403
import base64
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from urllib.parse import urlparse

from llama_stack.providers.datatypes import DatasetsProtocolPrivate
from llama_stack.providers.utils.datasetio.url_utils import get_dataframe_from_url
from llama_stack.providers.utils.kvstore import kvstore_impl

from .config import LocalFSDatasetIOConfig


DATASETS_PREFIX = "localfs_datasets:"


class BaseDataset(ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError()

    @abstractmethod
    def load(self):
        raise NotImplementedError()


@dataclass
class DatasetInfo:
    dataset_def: Dataset
    dataset_impl: BaseDataset


class PandasDataframeDataset(BaseDataset):
    def __init__(self, dataset_def: Dataset, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dataset_def = dataset_def
        self.df = None

    def __len__(self) -> int:
        assert self.df is not None, "Dataset not loaded. Please call .load() first"
        return len(self.df)

    def __getitem__(self, idx):
        assert self.df is not None, "Dataset not loaded. Please call .load() first"
        if isinstance(idx, slice):
            return self.df.iloc[idx].to_dict(orient="records")
        else:
            return self.df.iloc[idx].to_dict()

    def _validate_dataset_schema(self, df) -> pandas.DataFrame:
        # note that we will drop any columns in dataset that are not in the schema
        df = df[self.dataset_def.dataset_schema.keys()]
        # check all columns in dataset schema are present
        assert len(df.columns) == len(self.dataset_def.dataset_schema)
        # TODO: type checking against column types in dataset schema
        return df

    def load(self) -> None:
        if self.df is not None:
            return

        df = get_dataframe_from_url(self.dataset_def.url)
        if df is None:
            raise ValueError(f"Failed to load dataset from {self.dataset_def.url}")

        self.df = self._validate_dataset_schema(df)


class LocalFSDatasetIOImpl(DatasetIO, DatasetsProtocolPrivate):
    def __init__(self, config: LocalFSDatasetIOConfig) -> None:
        self.config = config
        # local registry for keeping track of datasets within the provider
        self.dataset_infos = {}
        self.kvstore = None

    async def initialize(self) -> None:
        self.kvstore = await kvstore_impl(self.config.kvstore)
        # Load existing datasets from kvstore
        start_key = DATASETS_PREFIX
        end_key = f"{DATASETS_PREFIX}\xff"
        stored_datasets = await self.kvstore.range(start_key, end_key)

        for dataset in stored_datasets:
            dataset = Dataset.model_validate_json(dataset)
            dataset_impl = PandasDataframeDataset(dataset)
            self.dataset_infos[dataset.identifier] = DatasetInfo(
                dataset_def=dataset,
                dataset_impl=dataset_impl,
            )

    async def shutdown(self) -> None: ...

    async def register_dataset(
        self,
        dataset: Dataset,
    ) -> None:
        # Store in kvstore
        key = f"{DATASETS_PREFIX}{dataset.identifier}"
        await self.kvstore.set(
            key=key,
            value=dataset.json(),
        )
        dataset_impl = PandasDataframeDataset(dataset)
        self.dataset_infos[dataset.identifier] = DatasetInfo(
            dataset_def=dataset,
            dataset_impl=dataset_impl,
        )

    async def unregister_dataset(self, dataset_id: str) -> None:
        key = f"{DATASETS_PREFIX}{dataset_id}"
        await self.kvstore.delete(key=key)
        del self.dataset_infos[dataset_id]

    async def get_rows_paginated(
        self,
        dataset_id: str,
        rows_in_page: int,
        page_token: Optional[str] = None,
        filter_condition: Optional[str] = None,
    ) -> PaginatedRowsResult:
        dataset_info = self.dataset_infos.get(dataset_id)
        dataset_info.dataset_impl.load()

        if page_token and not page_token.isnumeric():
            raise ValueError("Invalid page_token")

        if page_token is None or len(page_token) == 0:
            next_page_token = 0
        else:
            next_page_token = int(page_token)

        start = next_page_token
        if rows_in_page == -1:
            end = len(dataset_info.dataset_impl)
        else:
            end = min(start + rows_in_page, len(dataset_info.dataset_impl))

        rows = dataset_info.dataset_impl[start:end]

        return PaginatedRowsResult(
            rows=rows,
            total_count=len(rows),
            next_page_token=str(end),
        )

    async def append_rows(self, dataset_id: str, rows: List[Dict[str, Any]]) -> None:
        dataset_info = self.dataset_infos.get(dataset_id)
        if dataset_info is None:
            raise ValueError(f"Dataset with id {dataset_id} not found")

        dataset_impl = dataset_info.dataset_impl
        dataset_impl.load()

        new_rows_df = pandas.DataFrame(rows)
        new_rows_df = dataset_impl._validate_dataset_schema(new_rows_df)
        dataset_impl.df = pandas.concat(
            [dataset_impl.df, new_rows_df], ignore_index=True
        )

        url = str(dataset_info.dataset_def.url)
        parsed_url = urlparse(url)

        if parsed_url.scheme == "file" or not parsed_url.scheme:
            file_path = parsed_url.path
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            dataset_impl.df.to_csv(file_path, index=False)
        elif parsed_url.scheme == "data":
            # For data URLs, we need to update the base64-encoded content
            if not parsed_url.path.startswith("text/csv;base64,"):
                raise ValueError("Data URL must be a base64-encoded CSV")

            csv_buffer = dataset_impl.df.to_csv(index=False)
            base64_content = base64.b64encode(csv_buffer.encode("utf-8")).decode(
                "utf-8"
            )
            dataset_info.dataset_def.url = URL(
                uri=f"data:text/csv;base64,{base64_content}"
            )
        else:
            raise ValueError(
                f"Unsupported URL scheme: {parsed_url.scheme}. Only file:// and data: URLs are supported for writing."
            )
