# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from .datasets import CustomDataset

# TODO: make this into a config based registry
DATASETS_REGISTRY = {
    "mmlu_eval": CustomDataset(
        name="mmlu_eval",
        url="https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv",
    ),
}


def get_dataset(dataset_id: str):
    # get dataset concrete dataset implementation
    return DATASETS_REGISTRY[dataset_id]
