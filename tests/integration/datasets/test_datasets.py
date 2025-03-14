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


def test_register_unregister_dataset(llama_stack_client):
    response = llama_stack_client.datasets.list()
    assert isinstance(response, list)
    assert len(response) == 0
