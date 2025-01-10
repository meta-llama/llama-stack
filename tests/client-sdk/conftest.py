# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import os

import pytest

from llama_stack import LlamaStackAsLibraryClient
from llama_stack.providers.tests.env import get_env_or_fail
from llama_stack_client import LlamaStackClient


@pytest.fixture(scope="session")
def llama_stack_client():
    if os.environ.get("LLAMA_STACK_CONFIG"):
        client = LlamaStackAsLibraryClient(get_env_or_fail("LLAMA_STACK_CONFIG"))
        client.initialize()
    elif os.environ.get("LLAMA_STACK_BASE_URL"):
        client = LlamaStackClient(base_url=get_env_or_fail("LLAMA_STACK_BASE_URL"))
    else:
        raise ValueError("LLAMA_STACK_CONFIG or LLAMA_STACK_BASE_URL must be set")
    return client
