# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

from abc import abstractmethod
from typing import Dict, List

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_toolchain.agentic_system.api import *  # noqa: F403


class CustomTool:
    """
    Developers can define their custom tools that models can use
    by extending this class.

    Developers need to provide
        - name
        - description
        - params_definition
        - implement tool's behavior in `run_impl` method

    NOTE: The return of the `run` method needs to be json serializable
    """

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_description(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_params_definition(self) -> Dict[str, ToolParamDefinition]:
        raise NotImplementedError

    def get_instruction_string(self) -> str:
        return f"Use the function '{self.get_name()}' to: {self.get_description()}"

    def parameters_for_system_prompt(self) -> str:
        return json.dumps(
            {
                "name": self.get_name(),
                "description": self.get_description(),
                "parameters": {
                    name: definition.__dict__
                    for name, definition in self.get_params_definition().items()
                },
            }
        )

    def get_tool_definition(self) -> FunctionCallToolDefinition:
        return FunctionCallToolDefinition(
            function_name=self.get_name(),
            description=self.get_description(),
            parameters=self.get_params_definition(),
        )

    @abstractmethod
    async def run(self, messages: List[Message]) -> List[Message]:
        raise NotImplementedError


class SingleMessageCustomTool(CustomTool):
    """
    Helper class to handle custom tools that take a single message
    Extending this class and implementing the `run_impl` method will
    allow for the tool be called by the model and the necessary plumbing.
    """

    async def run(self, messages: List[CompletionMessage]) -> List[ToolResponseMessage]:
        assert len(messages) == 1, "Expected single message"

        message = messages[0]

        tool_call = message.tool_calls[0]

        try:
            response = await self.run_impl(**tool_call.arguments)
            response_str = json.dumps(response, ensure_ascii=False)
        except Exception as e:
            response_str = f"Error when running tool: {e}"

        message = ToolResponseMessage(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            content=response_str,
        )
        return [message]

    @abstractmethod
    async def run_impl(self, *args, **kwargs):
        raise NotImplementedError()
