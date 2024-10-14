# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# TODO: make these import config based
from llama_stack.apis.dataset import *  # noqa: F403
from ..registry import Registry
from .dataset import CustomDataset, HuggingfaceDataset


class DatasetRegistry(Registry[BaseDataset]):
    _REGISTRY: Dict[str, BaseDataset] = {}


DATASETS_REGISTRY = [
    CustomDataset(
        config=CustomDatasetDef(
            identifier="mmlu-simple-eval-en",
            url="https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv",
        )
    ),
    HuggingfaceDataset(
        config=HuggingfaceDatasetDef(
            identifier="hellaswag",
            dataset_name="hellaswag",
            kwargs={"split": "validation", "trust_remote_code": True},
        )
    ),
]

for d in DATASETS_REGISTRY:
    DatasetRegistry.register(d.dataset_id, d)
