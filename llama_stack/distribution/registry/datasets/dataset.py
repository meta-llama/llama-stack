# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import pandas
from datasets import Dataset, load_dataset

from llama_stack.apis.dataset import *  # noqa: F403


class CustomDataset(BaseDataset[DictSample]):
    def __init__(self, config: CustomDatasetDef) -> None:
        super().__init__()
        self.config = config
        self.dataset = None
        self.index = 0

    @property
    def dataset_id(self) -> str:
        return self.config.identifier

    def __iter__(self) -> Iterator[DictSample]:
        if not self.dataset:
            self.load()
        return (DictSample(data=x) for x in self.dataset)

    def __str__(self) -> str:
        return f"CustomDataset({self.config})"

    def __len__(self) -> int:
        if not self.dataset:
            self.load()
        return len(self.dataset)

    def load(self, n_samples: Optional[int] = None) -> None:
        if self.dataset:
            return

        # TODO: better support w/ data url
        if self.config.url.endswith(".csv"):
            df = pandas.read_csv(self.config.url)
        elif self.config.url.endswith(".xlsx"):
            df = pandas.read_excel(self.config.url)

        if n_samples is not None:
            df = df.sample(n=n_samples)

        self.dataset = Dataset.from_pandas(df)


class HuggingfaceDataset(BaseDataset[DictSample]):
    def __init__(self, config: HuggingfaceDatasetDef):
        super().__init__()
        self.config = config
        self.dataset = None

    @property
    def dataset_id(self) -> str:
        return self.config.identifier

    def __iter__(self) -> Iterator[DictSample]:
        if not self.dataset:
            self.load()
        return (DictSample(data=x) for x in self.dataset)

    def __str__(self):
        return f"HuggingfaceDataset({self.config})"

    def __len__(self):
        if not self.dataset:
            self.load()
        return len(self.dataset)

    def load(self):
        if self.dataset:
            return
        self.dataset = load_dataset(self.config.dataset_name, **self.config.kwargs)
