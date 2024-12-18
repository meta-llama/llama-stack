# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, get_type_hints, List, Optional, Type, TypeVar

from llama_models.llama3.api.datatypes import ToolPromptFormat
from llama_stack.apis.tools.tools import Tool, ToolParameter, ToolReturn

T = TypeVar("T")


class BaseTool(ABC):
    """Base class for all tools"""

    requires_api_key: bool = False

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @classmethod
    @abstractmethod
    def tool_id(cls) -> str:
        """Unique identifier for the tool"""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments"""
        pass

    @classmethod
    def get_provider_config_type(cls) -> Optional[Type[T]]:
        """Override to specify a Pydantic model for tool configuration"""
        return None

    @classmethod
    def get_tool_definition(cls) -> Tool:
        """Generate a Tool definition from the class implementation"""
        # Get execute method
        execute_method = cls.execute
        signature = inspect.signature(execute_method)
        docstring = execute_method.__doc__ or "No description available"

        # Extract parameters
        parameters: List[ToolParameter] = []
        type_hints = get_type_hints(execute_method)

        for name, param in signature.parameters.items():
            if name == "self":
                continue

            param_type = type_hints.get(name, Any).__name__
            required = param.default == param.empty
            default = None if param.default == param.empty else param.default

            parameters.append(
                ToolParameter(
                    name=name,
                    type_hint=param_type,
                    description=f"Parameter: {name}",  # Could be enhanced with docstring parsing
                    required=required,
                    default=default,
                )
            )

        # Extract return info
        return_type = type_hints.get("return", Any).__name__

        return Tool(
            identifier=cls.tool_id(),
            provider_resource_id=cls.tool_id(),
            name=cls.__name__,
            description=docstring,
            parameters=parameters,
            returns=ToolReturn(
                type_hint=return_type, description="Tool execution result"
            ),
            tool_prompt_format=ToolPromptFormat.json,
        )
