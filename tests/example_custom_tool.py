# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict

from llama_models.llama3.api.datatypes import ToolParamDefinition
from llama_stack.tools.custom.datatypes import SingleMessageCustomTool


class GetBoilingPointTool(SingleMessageCustomTool):
    """Tool to give boiling point of a liquid
    Returns the correct value for water in Celcius and Fahrenheit
    and returns -1 for other liquids

    """

    def get_name(self) -> str:
        return "get_boiling_point"

    def get_description(self) -> str:
        return "Get the boiling point of a imaginary liquids (eg. polyjuice)"

    def get_params_definition(self) -> Dict[str, ToolParamDefinition]:
        return {
            "liquid_name": ToolParamDefinition(
                param_type="string", description="The name of the liquid", required=True
            ),
            "celcius": ToolParamDefinition(
                param_type="boolean",
                description="Whether to return the boiling point in Celcius",
                required=False,
            ),
        }

    async def run_impl(self, liquid_name: str, celcius: bool = True) -> int:
        if liquid_name.lower() == "polyjuice":
            if celcius:
                return -100
            else:
                return -212
        else:
            return -1
