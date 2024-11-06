# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import io
from typing import List, Optional

import pandas
from llama_models.llama3.api.datatypes import *  # noqa: F403

from llama_stack.apis.datasetio import *  # noqa: F403
import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass
from urllib.parse import unquote

from llama_stack.providers.datatypes import DatasetsProtocolPrivate
from llama_stack.providers.utils.memory.vector_store import parse_data_url

from .config import MetaReferenceDatasetIOConfig


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
    dataset_def: DatasetDef
    dataset_impl: BaseDataset


class PandasDataframeDataset(BaseDataset):
    def __init__(self, dataset_def: DatasetDef, *args, **kwargs) -> None:
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

        # TODO: more robust support w/ data url
        if self.dataset_def.url.uri.endswith(".csv"):
            df = pandas.read_csv(self.dataset_def.url.uri)
        elif self.dataset_def.url.uri.endswith(".xlsx"):
            df = pandas.read_excel(self.dataset_def.url.uri)
        elif self.dataset_def.url.uri.startswith("data:"):
            parts = parse_data_url(self.dataset_def.url.uri)
            data = parts["data"]
            if parts["is_base64"]:
                data = base64.b64decode(data)
            else:
                data = unquote(data)
                encoding = parts["encoding"] or "utf-8"
                data = data.encode(encoding)

            mime_type = parts["mimetype"]
            mime_category = mime_type.split("/")[0]
            data_bytes = io.BytesIO(data)

            if mime_category == "text":
                df = pandas.read_csv(data_bytes)
            else:
                df = pandas.read_excel(data_bytes)
        else:
            raise ValueError(f"Unsupported file type: {self.dataset_def.url}")

        self.df = self._validate_dataset_schema(df)


class MetaReferenceDatasetIOImpl(DatasetIO, DatasetsProtocolPrivate):
    def __init__(self, config: MetaReferenceDatasetIOConfig) -> None:
        self.config = config
        # local registry for keeping track of datasets within the provider
        self.dataset_infos = {}

    async def initialize(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def register_dataset(
        self,
        dataset_def: DatasetDef,
    ) -> None:
        dataset_impl = PandasDataframeDataset(dataset_def)
        self.dataset_infos[dataset_def.identifier] = DatasetInfo(
            dataset_def=dataset_def,
            dataset_impl=dataset_impl,
        )

    async def list_datasets(self) -> List[DatasetDef]:
        return [i.dataset_def for i in self.dataset_infos.values()]

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
