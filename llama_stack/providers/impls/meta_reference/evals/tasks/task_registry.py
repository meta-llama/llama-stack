# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from .mmlu_task import MMLUTask

# TODO: make this into a config based registry
TASKS_REGISTRY = {
    "mmlu": MMLUTask,
}


def get_task(task_id: str, dataset):
    task_impl = TASKS_REGISTRY[task_id]
    return task_impl(dataset)
