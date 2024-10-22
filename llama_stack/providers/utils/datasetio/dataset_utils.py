# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from llama_stack.apis.datasetio import *  # noqa: F403


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
