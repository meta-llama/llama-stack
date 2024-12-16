# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import json
from typing import Dict, List

from llama_stack_client.lib.agents.custom_tool import CustomTool
from llama_stack_client.types import CompletionMessage, ToolResponseMessage
from llama_stack_client.types.tool_param_definition_param import (
    ToolParamDefinitionParam,
)


class GetBoilingPointTool(CustomTool):
    """Tool to give boiling point of a liquid
    Returns the correct value for water in Celcius and Fahrenheit
    and returns -1 for other liquids

    """

    def run(self, messages: List[CompletionMessage]) -> List[ToolResponseMessage]:
        assert len(messages) == 1, "Expected single message"

        message = messages[0]

        tool_call = message.tool_calls[0]

        try:
            response = self.run_impl(**tool_call.arguments)
            response_str = json.dumps(response, ensure_ascii=False)
        except Exception as e:
            response_str = f"Error when running tool: {e}"

        message = ToolResponseMessage(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            content=response_str,
            role="ipython",
        )
        return [message]

    def get_name(self) -> str:
        return "get_boiling_point"

    def get_description(self) -> str:
        return "Get the boiling point of a imaginary liquids (eg. polyjuice)"

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return {
            "liquid_name": ToolParamDefinitionParam(
                param_type="string", description="The name of the liquid", required=True
            ),
            "celcius": ToolParamDefinitionParam(
                param_type="boolean",
                description="Whether to return the boiling point in Celcius",
                required=False,
            ),
        }

    def run_impl(self, liquid_name: str, celcius: bool = True) -> int:
        if liquid_name.lower() == "polyjuice":
            if celcius:
                return -100
            else:
                return -212
        else:
            return -1
