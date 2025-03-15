# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any, Dict, List, Optional

import datasets as hf_datasets

from llama_stack.apis.datasetio import DatasetIO, IterrowsResponse
from llama_stack.apis.datasets import Dataset
from llama_stack.providers.datatypes import DatasetsProtocolPrivate
from llama_stack.providers.utils.datasetio.url_utils import get_dataframe_from_url
from llama_stack.providers.utils.kvstore import kvstore_impl

from .config import HuggingfaceDatasetIOConfig

DATASETS_PREFIX = "datasets:"


def load_hf_dataset(dataset_def: Dataset):
    if dataset_def.metadata.get("path", None):
        dataset = hf_datasets.load_dataset(**dataset_def.metadata)
    else:
        df = get_dataframe_from_url(dataset_def.url)

        if df is None:
            raise ValueError(f"Failed to load dataset from {dataset_def.url}")

        dataset = hf_datasets.Dataset.from_pandas(df)

    # drop columns not specified by schema
    if dataset_def.dataset_schema:
        dataset = dataset.select_columns(list(dataset_def.dataset_schema.keys()))

    return dataset


class HuggingfaceDatasetIOImpl(DatasetIO, DatasetsProtocolPrivate):
    def __init__(self, config: HuggingfaceDatasetIOConfig) -> None:
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
            self.dataset_infos[dataset.identifier] = dataset

    async def shutdown(self) -> None: ...

    async def register_dataset(
        self,
        dataset_def: Dataset,
    ) -> None:
        # Store in kvstore
        key = f"{DATASETS_PREFIX}{dataset_def.identifier}"
        await self.kvstore.set(
            key=key,
            value=dataset_def.json(),
        )
        self.dataset_infos[dataset_def.identifier] = dataset_def

    async def unregister_dataset(self, dataset_id: str) -> None:
        key = f"{DATASETS_PREFIX}{dataset_id}"
        await self.kvstore.delete(key=key)
        del self.dataset_infos[dataset_id]

    async def iterrows(
        self,
        dataset_id: str,
        start_index: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> IterrowsResponse:
        dataset_def = self.dataset_infos[dataset_id]
        loaded_dataset = load_hf_dataset(dataset_def)

        start_index = start_index or 0

        if limit is None or limit == -1:
            end = len(loaded_dataset)
        else:
            end = min(start_index + limit, len(loaded_dataset))

        rows = [loaded_dataset[i] for i in range(start_index, end)]

        return IterrowsResponse(
            data=rows,
            next_index=end if end < len(loaded_dataset) else None,
        )

    async def append_rows(self, dataset_id: str, rows: List[Dict[str, Any]]) -> None:
        dataset_def = self.dataset_infos[dataset_id]
        loaded_dataset = load_hf_dataset(dataset_def)

        # Convert rows to HF Dataset format
        new_dataset = hf_datasets.Dataset.from_list(rows)

        # Concatenate the new rows with existing dataset
        updated_dataset = hf_datasets.concatenate_datasets([loaded_dataset, new_dataset])

        if dataset_def.metadata.get("path", None):
            updated_dataset.push_to_hub(dataset_def.metadata["path"])
        else:
            raise NotImplementedError("Uploading to URL-based datasets is not supported yet")
