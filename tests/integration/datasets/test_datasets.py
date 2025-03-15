# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import mimetypes
import os
from pathlib import Path

import pytest

# How to run this test:
#
# LLAMA_STACK_CONFIG="template-name" pytest -v tests/integration/datasets


def test_register_dataset(llama_stack_client):
    dataset = llama_stack_client.datasets.register(
        purpose="eval/messages-answer",
        source={
            "type": "uri",
            "uri": "huggingface://datasets/llamastack/simpleqa?split=train",
        },
    )
    assert dataset.identifier is not None
    assert dataset.provider_id == "huggingface"
    iterrow_response = llama_stack_client.datasets.iterrows(
        dataset.identifier, limit=10
    )
    assert len(iterrow_response.data) == 10
    assert iterrow_response.next_index is not None
