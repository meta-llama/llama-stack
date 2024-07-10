import requests
from dataclasses import dataclass, field, asdict
from typing import List, Set, Optional, Union, Protocol
from enum import Enum

import json

from model_types import * 
from agentic_system_types import *
from api_definitions import * 

class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class AgenticSystemClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def create_agentic_system(self, request: AgenticSystemCreateRequest) -> AgenticSystemCreateResponse:
        response = requests.post(f"{self.base_url}/agentic_system/create", data=json.dumps(asdict(request), cls=EnumEncoder), headers={'Content-Type': 'application/json'})
        response.raise_for_status()
        return AgenticSystemCreateResponse(**response.json())

    def execute_agentic_system(self, request: AgenticSystemExecuteRequest) -> Union[AgenticSystemExecuteResponse, AgenticSystemExecuteResponseStreamChunk]:
        response = requests.post(f"{self.base_url}/agentic_system/execute", data=json.dumps(asdict(request), cls=EnumEncoder), headers={'Content-Type': 'application/json'})
        response.raise_for_status()
        response_json = response.json()
        if 'turn' in response_json:
            return AgenticSystemExecuteResponse(**response_json)
        else:
            return AgenticSystemExecuteResponseStreamChunk(**response_json)

# Example usage
if __name__ == "__main__":
    client = AgenticSystemClient("http://localhost:5000")

    # Create a new agentic system
    create_request = AgenticSystemCreateRequest(
        instructions="Your instructions here",
        model=InstructModel.llama3_8b_chat,
    )
    create_response = client.create_agentic_system(create_request)
    print("Agent ID:", create_response.agent_id)

    # Execute the agentic system
    execute_request = AgenticSystemExecuteRequest(
        agent_id=create_response.agent_id,
        messages=[Message(role="user", content="Tell me a joke")],
        turn_history=[],
        stream=False
    )
    execute_response = client.execute_agentic_system(execute_request)
    print("Execute Response:", execute_response)
