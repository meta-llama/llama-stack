# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.distribution.datatypes import (
    AdapterSpec,
    Api,
    InlineProviderSpec,
    ProviderSpec,
    remote_provider_spec,
)


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.tool_runtime,
            provider_type="inline::memory-runtime",
            pip_packages=[],
            module="llama_stack.providers.inline.tool_runtime.memory",
            config_class="llama_stack.providers.inline.tool_runtime.memory.config.MemoryToolConfig",
            api_dependencies=[Api.memory, Api.memory_banks, Api.inference],
        ),
        InlineProviderSpec(
            api=Api.tool_runtime,
            provider_type="inline::code-interpreter",
            pip_packages=[],
            module="llama_stack.providers.inline.tool_runtime.code_interpreter",
            config_class="llama_stack.providers.inline.tool_runtime.code_interpreter.config.CodeInterpreterToolConfig",
        ),
        remote_provider_spec(
            api=Api.tool_runtime,
            adapter=AdapterSpec(
                adapter_type="brave-search",
                module="llama_stack.providers.remote.tool_runtime.brave_search",
                config_class="llama_stack.providers.remote.tool_runtime.brave_search.config.BraveSearchToolConfig",
                pip_packages=["requests"],
                provider_data_validator="llama_stack.providers.remote.tool_runtime.brave_search.BraveSearchToolProviderDataValidator",
            ),
        ),
        remote_provider_spec(
            api=Api.tool_runtime,
            adapter=AdapterSpec(
                adapter_type="tavily-search",
                module="llama_stack.providers.remote.tool_runtime.tavily_search",
                config_class="llama_stack.providers.remote.tool_runtime.tavily_search.config.TavilySearchToolConfig",
                pip_packages=["requests"],
                provider_data_validator="llama_stack.providers.remote.tool_runtime.tavily_search.TavilySearchToolProviderDataValidator",
            ),
        ),
        remote_provider_spec(
            api=Api.tool_runtime,
            adapter=AdapterSpec(
                adapter_type="model-context-protocol",
                module="llama_stack.providers.remote.tool_runtime.model_context_protocol",
                config_class="llama_stack.providers.remote.tool_runtime.model_context_protocol.config.ModelContextProtocolConfig",
                pip_packages=["mcp"],
            ),
        ),
    ]
