# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import pytest

from llama_stack.providers.tests.env import get_env_or_fail
from llama_stack_client import LlamaStackClient


@pytest.fixture
def llama_stack_client():
    """Fixture to create a fresh LlamaStackClient instance for each test"""
    return LlamaStackClient(base_url=get_env_or_fail("LLAMA_STACK_BASE_URL"))
