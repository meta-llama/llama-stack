# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from urllib.parse import parse_qs, urlparse

import pandas
from datasets import Dataset, load_dataset


class BaseDataset:
    def __init__(self, name: str):
        self.dataset = None
        self.dataset_id = name
        self.type = self.__class__.__name__


class CustomDataset(BaseDataset):
    def __init__(self, name, url):
        super().__init__(name)
        self.url = url
        df = pandas.read_csv(self.url)
        self.dataset = Dataset.from_pandas(df)


class HFDataset(BaseDataset):
    def __init__(self, name, url):
        super().__init__(name)
        # URL following OpenAI's evals - hf://hendrycks_test?name=business_ethics&split=validation
        self.url = url
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        query = {k: v[0] for k, v in query.items()}

        if parsed.scheme != "hf":
            raise ValueError(f"Unknown HF dataset: {url}")

        self.dataset = load_dataset(path, **query)
