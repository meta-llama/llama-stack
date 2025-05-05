# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import socket
import threading
import time

import httpx
import mcp.types as types
import pytest
import uvicorn
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route


@pytest.fixture(scope="module")
def mcp_server():
    server = FastMCP("FastMCP Test Server")

    @server.tool()
    async def fetch(url: str, ctx: Context) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        headers = {"User-Agent": "MCP Test Server (github.com/modelcontextprotocol/python-sdk)"}
        async with httpx.AsyncClient(follow_redirects=True, headers=headers) as client:
            response = await client.get(url)
            response.raise_for_status()
            return [types.TextContent(type="text", text=response.text)]

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            await server._mcp_server.run(
                streams[0],
                streams[1],
                server._mcp_server.create_initialization_options(),
            )

    app = Starlette(
        debug=True,
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

    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=port)

    # Start the server in a new thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Polling until the server is ready
    timeout = 10
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = httpx.get(f"http://localhost:{port}/sse")
            if response.status_code == 200:
                break
        except (httpx.RequestError, httpx.HTTPStatusError):
            pass
        time.sleep(0.1)

    yield port


def test_register_and_unregister_toolgroup(llama_stack_client, mcp_server):
    """
    Integration test for registering and unregistering a toolgroup using the ToolGroups API.
    """
    port = mcp_server
    test_toolgroup_id = "remote::web-fetch"
    provider_id = "model-context-protocol"

    # Cleanup before running the test
    toolgroups = llama_stack_client.toolgroups.list()
    for toolgroup in toolgroups:
        if toolgroup.identifier == test_toolgroup_id:
            llama_stack_client.toolgroups.unregister(toolgroup_id=test_toolgroup_id)

    # Register the toolgroup
    llama_stack_client.toolgroups.register(
        toolgroup_id=test_toolgroup_id,
        provider_id=provider_id,
        mcp_endpoint=dict(uri=f"http://localhost:{port}/sse"),
    )

    # Verify registration
    registered_toolgroup = llama_stack_client.toolgroups.get(toolgroup_id=test_toolgroup_id)
    assert registered_toolgroup is not None
    assert registered_toolgroup.identifier == test_toolgroup_id
    assert registered_toolgroup.provider_id == provider_id

    # Verify tools listing
    tools_list_response = llama_stack_client.tools.list(toolgroup_id=test_toolgroup_id)
    assert isinstance(tools_list_response, list)
    assert tools_list_response

    # Unregister the toolgroup
    llama_stack_client.toolgroups.unregister(toolgroup_id=test_toolgroup_id)

    # Verify it is unregistered
    with pytest.raises(Exception, match=f"Tool group '{test_toolgroup_id}' not found"):
        llama_stack_client.toolgroups.get(toolgroup_id=test_toolgroup_id)

    # Verify tools are also unregistered
    unregister_tools_list_response = llama_stack_client.tools.list(toolgroup_id=test_toolgroup_id)
    assert isinstance(unregister_tools_list_response, list)
    assert not unregister_tools_list_response
