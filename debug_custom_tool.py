# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from typing import Dict, List
from uuid import uuid4

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.client_tool import ClientTool
from llama_stack_client.types import ToolResponseMessage
from llama_stack_client.types.shared.completion_message import CompletionMessage
from llama_stack_client.types.shared_params.agent_config import AgentConfig, ToolConfig
from llama_stack_client.types.tool_def_param import Parameter
from rich.pretty import pprint


class TestClientTool(ClientTool):
    """Tool to give boiling point of a liquid
    Returns the correct value for polyjuice in Celcius and Fahrenheit
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
            role="tool",
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            content=response_str,
        )
        return message

    def get_name(self) -> str:
        return "get_boiling_point"

    def get_description(self) -> str:
        return "Get the boiling point of imaginary liquids (eg. polyjuice)"

    def get_params_definition(self) -> Dict[str, Parameter]:
        return {
            "liquid_name": Parameter(
                name="liquid_name",
                parameter_type="string",
                description="The name of the liquid",
                required=True,
            ),
            "celcius": Parameter(
                name="celcius",
                parameter_type="boolean",
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


if __name__ == "__main__":
    tool = TestClientTool()
    agent_config = AgentConfig(
        model="meta-llama/Llama-3.1-8B-Instruct",
        instructions="You are a helpful assistant",
        sampling_params={
            "strategy": {
                "type": "top_p",
                "temperature": 1.0,
                "top_p": 0.9,
            },
        },
        toolgroups=[],
        input_shields=[],
        output_shields=[],
        tool_config=ToolConfig(
            tool_choice="auto",
            tool_prompt_format="json",
        ),
        client_tools=[tool.get_tool_definition()],
        enable_session_persistence=False,
    )
    client = LlamaStackClient(base_url="http://localhost:8321")
    agent = Agent(client, agent_config, client_tools=(tool,))
    session_id = agent.create_session(f"test-session-{uuid4()}")
    simple_hello = agent.create_turn(
        messages=[
            {
                "role": "user",
                "content": "What is the boiling point of polyjuice in Celcius?",
            }
        ],
        session_id=session_id,
    )
    for chunk in simple_hello:
        pprint(chunk)
