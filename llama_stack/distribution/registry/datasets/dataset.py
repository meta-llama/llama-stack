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

    def __iter__(self) -> Iterator[DictSample]:
        return self

    def __next__(self) -> DictSample:
        if not self.dataset:
            self.load()
        if self.index >= len(self.dataset):
            raise StopIteration
        sample = DictSample(data=self.dataset[self.index])
        self.index += 1
        return sample

    def __str__(self):
        return f"CustomDataset({self.config})"

    def __len__(self):
        if not self.dataset:
            self.load()
        return len(self.dataset)

    def load(self):
        if self.dataset:
            return
        # TODO: better support w/ data url
        if self.config.url.endswith(".csv"):
            df = pandas.read_csv(self.config.url)
        elif self.config.url.endswith(".xlsx"):
            df = pandas.read_excel(self.config.url)

        self.dataset = Dataset.from_pandas(df)


class HuggingfaceDataset(BaseDataset[DictSample]):
    def __init__(self, config: HuggingfaceDatasetDef):
        super().__init__()
        self.config = config
        self.dataset = None
        self.index = 0

    def __iter__(self) -> Iterator[DictSample]:
        return self

    def __next__(self) -> DictSample:
        if not self.dataset:
            self.load()
        if self.index >= len(self.dataset):
            raise StopIteration
        sample = DictSample(data=self.dataset[self.index])
        self.index += 1
        return sample

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
        # parsed = urlparse(self.url)

        # if parsed.scheme != "hf":
        #     raise ValueError(f"Unknown HF dataset: {self.url}")

        # query = parse_qs(parsed.query)
        # query = {k: v[0] for k, v in query.items()}
        # path = parsed.netloc
        # self.dataset = load_dataset(path, **query)
