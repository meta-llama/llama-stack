# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import socket
import threading
import time

import httpx
import mcp.types as types
import pytest
import uvicorn
from llama_stack_client import Agent
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.routing import Mount, Route

from llama_stack import LlamaStackAsLibraryClient
from llama_stack.distribution.datatypes import AuthenticationRequiredError

AUTH_TOKEN = "test-token"


@pytest.fixture(scope="module")
def mcp_server():
    server = FastMCP("FastMCP Test Server")

    @server.tool()
    async def greet_everyone(
        url: str, ctx: Context
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        return [types.TextContent(type="text", text="Hello, world!")]

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        auth_header = request.headers.get("Authorization")
        auth_token = None
        if auth_header and auth_header.startswith("Bearer "):
            auth_token = auth_header.split(" ")[1]

        if auth_token != AUTH_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized")

        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            await server._mcp_server.run(
                streams[0],
                streams[1],
                server._mcp_server.create_initialization_options(),
            )
            return Response()

    app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

    def get_open_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    port = get_open_port()

    config = uvicorn.Config(app, host="0.0.0.0", port=port)
    server_instance = uvicorn.Server(config)
    app.state.uvicorn_server = server_instance

    def run_server():
        server_instance.run()

    # Start the server in a new thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Polling until the server is ready
    timeout = 10
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = httpx.get(f"http://localhost:{port}/sse")
            if response.status_code == 401:
                break
        except httpx.RequestError:
            pass
        time.sleep(0.1)

    yield port

    # Tell server to exit
    server_instance.should_exit = True
    server_thread.join(timeout=5)


def test_mcp_invocation(llama_stack_client, mcp_server):
    port = mcp_server
    test_toolgroup_id = "remote::mcptest"

    # registering itself should fail since it requires listing tools
    with pytest.raises(Exception, match="Unauthorized"):
        llama_stack_client.toolgroups.register(
            toolgroup_id=test_toolgroup_id,
            provider_id="model-context-protocol",
            mcp_endpoint=dict(uri=f"http://localhost:{port}/sse"),
        )

    provider_data = {
        "mcp_headers": {
            f"http://localhost:{port}/sse": [
                f"Authorization: Bearer {AUTH_TOKEN}",
            ],
        },
    }
    auth_headers = {
        "X-LlamaStack-Provider-Data": json.dumps(provider_data),
    }

    try:
        llama_stack_client.toolgroups.unregister(toolgroup_id=test_toolgroup_id, extra_headers=auth_headers)
    except Exception as e:
        # An error is OK since the toolgroup may not exist
        print(f"Error unregistering toolgroup: {e}")

    llama_stack_client.toolgroups.register(
        toolgroup_id=test_toolgroup_id,
        provider_id="model-context-protocol",
        mcp_endpoint=dict(uri=f"http://localhost:{port}/sse"),
        extra_headers=auth_headers,
    )
    response = llama_stack_client.tools.list(
        toolgroup_id=test_toolgroup_id,
        extra_headers=auth_headers,
    )
    assert len(response) == 1
    assert response[0].identifier == "greet_everyone"
    assert response[0].type == "tool"
    assert len(response[0].parameters) == 1
    p = response[0].parameters[0]
    assert p.name == "url"
    assert p.parameter_type == "string"
    assert p.required

    response = llama_stack_client.tool_runtime.invoke_tool(
        tool_name=response[0].identifier,
        kwargs=dict(url="https://www.google.com"),
        extra_headers=auth_headers,
    )
    content = response.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert content[0].text == "Hello, world!"

    models = llama_stack_client.models.list()
    model_id = models[0].identifier
    print(f"Using model: {model_id}")
    agent = Agent(
        client=llama_stack_client,
        model=model_id,
        instructions="You are a helpful assistant.",
        tools=[test_toolgroup_id],
    )
    session_id = agent.create_session("test-session")
    response = agent.create_turn(
        session_id=session_id,
        messages=[
            {
                "role": "user",
                "content": "Yo. Use tools.",
            }
        ],
        stream=False,
        extra_headers=auth_headers,
    )
    steps = response.steps
    first = steps[0]
    assert first.step_type == "inference"
    assert len(first.api_model_response.tool_calls) == 1
    tool_call = first.api_model_response.tool_calls[0]
    assert tool_call.tool_name == "greet_everyone"

    second = steps[1]
    assert second.step_type == "tool_execution"
    tool_response_content = second.tool_responses[0].content
    assert len(tool_response_content) == 1
    assert tool_response_content[0].type == "text"
    assert tool_response_content[0].text == "Hello, world!"

    third = steps[2]
    assert third.step_type == "inference"
    assert len(third.api_model_response.tool_calls) == 0

    # when streaming, we currently don't check auth headers upfront and fail the request
    # early. but we should at least be generating a 401 later in the process.
    response = agent.create_turn(
        session_id=session_id,
        messages=[
            {
                "role": "user",
                "content": "Yo. Use tools.",
            }
        ],
        stream=True,
    )
    if isinstance(llama_stack_client, LlamaStackAsLibraryClient):
        with pytest.raises(AuthenticationRequiredError):
            for _ in response:
                pass
    else:
        error_chunks = [chunk for chunk in response if "error" in chunk.model_dump()]
        assert len(error_chunks) == 1
        chunk = error_chunks[0].model_dump()
        assert "Unauthorized" in chunk["error"]["message"]
