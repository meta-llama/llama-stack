# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.datasets import *  # noqa: F403
from llama_stack.distribution.registry.datasets import DatasetRegistry
from llama_stack.distribution.registry.datasets.dataset_wrappers import (
    CustomDataset,
    HuggingfaceDataset,
)


class DatasetRegistryImpl(Datasets):
    """API Impl to interact with underlying dataset registry"""

    def __init__(
        self,
    ) -> None:
        pass

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def create_dataset(
        self,
        dataset_def: DatasetDef,
    ) -> CreateDatasetResponse:
        if dataset_def.type == DatasetType.huggingface.value:
            dataset_cls = HuggingfaceDataset(dataset_def)
        else:
            dataset_cls = CustomDataset(dataset_def)

        try:
            DatasetRegistry.register(
                dataset_def.identifier,
                dataset_cls,
            )
        except ValueError as e:
            return CreateDatasetResponse(
                status=DatasetsResponseStatus.fail,
                msg=str(e),
            )

        return CreateDatasetResponse(
            status=DatasetsResponseStatus.success,
            msg=f"Dataset '{dataset_def.identifier}' registered",
        )

    async def get_dataset(
        self,
        dataset_identifier: str,
    ) -> Optional[DatasetDef]:
        try:
            dataset_ref = DatasetRegistry.get(dataset_identifier).config
        except ValueError as e:
            return None

        return dataset_ref

    async def delete_dataset(self, dataset_identifier: str) -> DeleteDatasetResponse:
        try:
            DatasetRegistry.delete(dataset_identifier)
        except ValueError as e:
            return DeleteDatasetResponse(
                status=DatasetsResponseStatus.fail,
                msg=str(e),
            )

        return DeleteDatasetResponse(
            status=DatasetsResponseStatus.success,
            msg=f"Dataset '{dataset_identifier}' deleted",
        )

    async def list_datasets(self) -> List[DatasetDef]:
        return [
            DatasetRegistry.get(dataset_identifier).config
            for dataset_identifier in DatasetRegistry.names()
        ]
