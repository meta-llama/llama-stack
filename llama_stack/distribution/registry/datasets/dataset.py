# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# from llama_stack.apis.datasets import *
# from llama_stack.distribution.registry.datasets import DatasetRegistry  # noqa: F403
# from ..registry import Registry
# from .dataset_wrappers import CustomDataset, HuggingfaceDataset


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
    ) -> None:
        print(f"Creating dataset {dataset.identifier}")

    async def get_dataset(
        self,
        dataset_identifier: str,
    ) -> DatasetDef:
        pass

    async def delete_dataset(self, dataset_identifier: str) -> None:
        pass
