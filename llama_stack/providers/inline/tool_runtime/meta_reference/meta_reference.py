# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
from enum import Enum
from typing import Any, Dict

import llama_stack.providers.inline.tool_runtime.meta_reference.builtins as builtins

from llama_stack.apis.tools import Tool, ToolRuntime
from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.providers.datatypes import ToolsProtocolPrivate

from .config import MetaReferenceToolRuntimeConfig

logger = logging.getLogger(__name__)


class ToolType(Enum):
    bing_search = "bing_search"
    brave_search = "brave_search"
    tavily_search = "tavily_search"
    print_tool = "print_tool"


class MetaReferenceToolRuntimeImpl(
    ToolsProtocolPrivate, ToolRuntime, NeedsRequestProviderData
):
    def __init__(self, config: MetaReferenceToolRuntimeConfig):
        self.config = config

    async def initialize(self):
        pass

    async def register_tool(self, tool: Tool):
        print(f"registering tool {tool.identifier}")
        if tool.provider_resource_id not in ToolType.__members__:
            raise ValueError(
                f"Tool {tool.identifier} not a supported tool by Meta Reference"
            )

    async def unregister_tool(self, tool_id: str) -> None:
        raise NotImplementedError("Meta Reference does not support unregistering tools")

    async def invoke_tool(self, tool_id: str, args: Dict[str, Any]) -> Any:
        tool = await self.tool_store.get_tool(tool_id)
        if args.get("__api_key__") is not None:
            logger.warning(
                "__api_key__ is a reserved argument for this tool: {tool_id}"
            )
        args["__api_key__"] = self._get_api_key()
        return await getattr(builtins, tool.provider_resource_id)(**args)

    def _get_api_key(self) -> str:
        provider_data = self.get_request_provider_data()
        if provider_data is None or not provider_data.api_key:
            raise ValueError(
                'Pass Search provider\'s API Key in the header X-LlamaStack-ProviderData as { "api_key": <your api key>}'
            )
        return provider_data.api_key
