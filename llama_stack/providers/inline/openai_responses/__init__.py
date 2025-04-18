# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from llama_stack.apis.datatypes import Api

from .config import OpenAIResponsesImplConfig


async def get_provider_impl(config: OpenAIResponsesImplConfig, deps: Dict[Api, Any]):
    from .openai_responses import OpenAIResponsesImpl

    impl = OpenAIResponsesImpl(
        config, deps[Api.models], deps[Api.inference], deps[Api.tool_groups], deps[Api.tool_runtime]
    )
    await impl.initialize()
    return impl
