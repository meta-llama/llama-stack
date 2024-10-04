# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from .datasets import CustomDataset, HFDataset

# TODO: make this into a config based registry
DATASETS_REGISTRY = {
    "mmlu-simple-eval-en": CustomDataset(
        name="mmlu_eval",
        url="https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv",
    ),
    "mmmu-accounting": HFDataset(
        name="mmlu_eval",
        url="hf://hellaswag?split=validation&trust_remote_code=True",
    ),
}


def get_dataset(dataset_id: str):
    dataset = DATASETS_REGISTRY[dataset_id]
    dataset.load()
    return dataset
