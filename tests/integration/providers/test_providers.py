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
    def test_providers(self, llama_stack_client: LlamaStackAsLibraryClient | LlamaStackClient):
        provider_list = llama_stack_client.providers.list()
        assert provider_list is not None
        assert len(provider_list) > 0

        for provider in provider_list:
            pid = provider.provider_id
            provider = llama_stack_client.providers.retrieve(pid)
            assert provider is not None

    @pytest.mark.asyncio
    def test_providers_update(self, llama_stack_client: LlamaStackAsLibraryClient | LlamaStackClient):
        new_cfg = {"url": "http://localhost:12345"}

        _ = llama_stack_client.providers.retrieve("ollama")

        llama_stack_client.providers.update(
            api="inference",
            provider_id="ollama",
            provider_type="remote::ollama",
            config=new_cfg,
        )

        new_provider = llama_stack_client.providers.retrieve("ollama")

        assert new_provider.config == new_cfg
