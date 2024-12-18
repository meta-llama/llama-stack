# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import importlib
import logging
import pkgutil
from typing import Any, Dict, Optional, Type

from llama_stack.apis.tools import Tool, ToolRuntime
from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.providers.datatypes import ToolsProtocolPrivate
from llama_stack.providers.inline.tool_runtime.meta_reference.tools.base import BaseTool

from .config import MetaReferenceToolRuntimeConfig

logger = logging.getLogger(__name__)


class MetaReferenceToolRuntimeImpl(
    ToolsProtocolPrivate, ToolRuntime, NeedsRequestProviderData
):
    def __init__(self, config: MetaReferenceToolRuntimeConfig):
        self.config = config
        self.tools: Dict[str, Type[BaseTool]] = {}
        self._discover_tools()

    def _discover_tools(self):
        # Import all tools from the tools package
        tools_package = "llama_stack.providers.inline.tool_runtime.tools"
        package = importlib.import_module(tools_package)

        for _, name, _ in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"{tools_package}.{name}")
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseTool)
                    and attr != BaseTool
                ):
                    self.tools[attr.tool_id()] = attr

    async def initialize(self):
        pass

    async def register_tool(self, tool: Tool):
        if tool.identifier not in self.tools:
            raise ValueError(f"Tool {tool.identifier} not found in available tools")

        # Validate provider_metadata against tool's config type if specified
        tool_class = self.tools[tool.identifier]
        config_type = tool_class.get_provider_config_type()
        if (
            config_type
            and tool.provider_metadata
            and tool.provider_metadata.get("config")
        ):
            config_type(**tool.provider_metadata.get("config"))

    async def invoke_tool(self, tool_id: str, args: Dict[str, Any]) -> Any:
        if tool_id not in self.tools:
            raise ValueError(f"Tool {tool_id} not found")

        tool_instance = await self._create_tool_instance(tool_id)
        return await tool_instance.execute(**args)

    async def unregister_tool(self, tool_id: str) -> None:
        raise NotImplementedError("Meta Reference does not support unregistering tools")

    async def _create_tool_instance(
        self, tool_id: str, tool_def: Optional[Tool] = None
    ) -> BaseTool:
        """Create a new tool instance with proper configuration"""
        if tool_id not in self.tools:
            raise ValueError(f"Tool {tool_id} not found in available tools")

        tool_class = self.tools[tool_id]

        # Get tool definition if not provided
        if tool_def is None:
            tool_def = await self.tool_store.get_tool(tool_id)

        # Build configuration
        config = dict(tool_def.provider_metadata.get("config") or {})
        if tool_class.requires_api_key:
            config["api_key"] = self._get_api_key()

        return tool_class(config=config)

    def _get_api_key(self) -> str:
        provider_data = self.get_request_provider_data()
        if provider_data is None or not provider_data.api_key:
            raise ValueError(
                'Pass Search provider\'s API Key in the header X-LlamaStack-ProviderData as { "api_key": <your api key>}'
            )
        return provider_data.api_key
