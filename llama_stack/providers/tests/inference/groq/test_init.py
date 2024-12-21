# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from llama_stack.apis.inference import Inference
from llama_stack.providers.remote.inference.groq import get_adapter_impl
from llama_stack.providers.remote.inference.groq.config import GroqConfig
from llama_stack.providers.remote.inference.groq.groq import GroqInferenceAdapter

from llama_stack.providers.remote.inference.ollama import OllamaImplConfig


class TestGroqInit:
    @pytest.mark.asyncio
    async def test_raises_runtime_error_if_config_is_not_groq_config(self):
        config = OllamaImplConfig(model="llama3.1-8b-8192")

        with pytest.raises(RuntimeError):
            await get_adapter_impl(config, None)

    @pytest.mark.asyncio
    async def test_returns_groq_adapter(self):
        config = GroqConfig()
        adapter = await get_adapter_impl(config, None)
        assert type(adapter) is GroqInferenceAdapter
        assert isinstance(adapter, Inference)
