# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import List, Optional

import pandas

from llama_models.llama3.api.datatypes import *  # noqa: F403

from llama_stack.apis.datasetio import *  # noqa: F403
from dataclasses import dataclass

from llama_stack.providers.datatypes import DatasetsProtocolPrivate
from llama_stack.providers.utils.datasetio.dataset_utils import BaseDataset

from .config import MetaReferenceDatasetIOConfig


@dataclass
class DatasetInfo:
    dataset_def: DatasetDef
    dataset_impl: BaseDataset
    next_page_token: Optional[int] = None


class CustomDataset(BaseDataset):
    def __init__(self, dataset_def: DatasetDef, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dataset_def = dataset_def
        # TODO: validate dataset_def against schema
        self.df = None
        self.load()

    def __len__(self) -> int:
        if self.df is None:
            self.load()
        return len(self.df)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.df.iloc[idx].to_dict(orient="records")
        else:
            return self.df.iloc[idx].to_dict()

    def load(self) -> None:
        if self.df:
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

        self.df = df


class MetaReferenceDatasetioImpl(DatasetIO, DatasetsProtocolPrivate):
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
        self.dataset_infos[dataset_def.identifier] = DatasetInfo(
            dataset_def=dataset_def,
            dataset_impl=CustomDataset(dataset_def),
            next_page_token=0,
        )

    async def list_datasets(self) -> List[DatasetDef]:
        return [i.dataset_def for i in self.dataset_infos.values()]

    async def get_rows_paginated(
        self,
        dataset_id: str,
        rows_in_page: int,
        page_token: Optional[int] = None,
        filter_condition: Optional[str] = None,
    ) -> PaginatedRowsResult:
        dataset_info = self.dataset_infos.get(dataset_id)
        if page_token is None:
            dataset_info.next_page_token = 0

        if rows_in_page == -1:
            rows = dataset_info.dataset_impl[dataset_info.next_page_token :]

        start = dataset_info.next_page_token
        end = min(start + rows_in_page, len(dataset_info.dataset_impl))
        rows = dataset_info.dataset_impl[start:end]
        dataset_info.next_page_token = end

        return PaginatedRowsResult(
            rows=rows,
            total_count=len(rows),
            next_page_token=dataset_info.next_page_token,
        )
