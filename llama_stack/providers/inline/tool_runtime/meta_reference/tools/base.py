# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar

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
