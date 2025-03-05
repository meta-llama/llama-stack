# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from llama_stack_client import LlamaStackClient

from llama_stack import LlamaStackAsLibraryClient


class TestProviders:
    @pytest.mark.asyncio
    def test_list(self, llama_stack_client: LlamaStackAsLibraryClient | LlamaStackClient):
        provider_list = llama_stack_client.providers.list()
        assert provider_list is not None
