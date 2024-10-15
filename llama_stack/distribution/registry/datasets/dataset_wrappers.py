# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import io

import pandas
from datasets import Dataset, load_dataset

from llama_stack.apis.datasets import *  # noqa: F403
from llama_stack.providers.utils.memory.vector_store import parse_data_url


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

        # TODO: more robust support w/ data url
        if self.config.url.endswith(".csv"):
            df = pandas.read_csv(self.config.url)
        elif self.config.url.endswith(".xlsx"):
            df = pandas.read_excel(self.config.url)
        elif self.config.url.startswith("data:"):
            parts = parse_data_url(self.config.url)
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
            raise ValueError(f"Unsupported file type: {self.config.url}")

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

    def load(self, n_samples: Optional[int] = None):
        if self.dataset:
            return

        if self.config.dataset_name:
            self.config.kwargs["name"] = self.config.dataset_name

        self.dataset = load_dataset(self.config.dataset_path, **self.config.kwargs)

        if n_samples:
            self.dataset = self.dataset.select(range(n_samples))

        if self.config.rename_columns_map:
            for k, v in self.config.rename_columns_map.items():
                self.dataset = self.dataset.rename_column(k, v)
