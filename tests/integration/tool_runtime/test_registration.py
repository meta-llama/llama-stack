# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import threading
import time

import httpx
import mcp.types as types
import pytest
import uvicorn
from llama_stack_client.types.shared_params.url import URL
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

    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=8000)

    # Start the server in a new thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Polling until the server is ready
    timeout = 10
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = httpx.get("http://localhost:8000/sse")
            if response.status_code == 200:
                break
        except (httpx.RequestError, httpx.HTTPStatusError):
            pass
        time.sleep(0.1)

    yield


def test_register_and_unregister_toolgroup(client_with_models, mcp_server):
    """
    Integration test for registering and unregistering a toolgroup using the ToolGroups API.
    """
    test_toolgroup_id = "remote::web-fetch"
    provider_id = "model-context-protocol"

    # Cleanup before running the test
    test_toolgroup = client_with_models.toolgroups.get(toolgroup_id=test_toolgroup_id)
    if test_toolgroup is not None:
        client_with_models.toolgroups.unregister(toolgroup_id=test_toolgroup_id)

    # Register the toolgroup
    client_with_models.toolgroups.register(
        toolgroup_id=test_toolgroup_id,
        provider_id=provider_id,
        mcp_endpoint=URL(uri="http://localhost:8000/sse"),
    )

    # Verify registration
    registered_toolgroup = client_with_models.toolgroups.get(toolgroup_id=test_toolgroup_id)
    assert registered_toolgroup is not None
    assert registered_toolgroup.identifier == test_toolgroup_id
    assert registered_toolgroup.provider_id == provider_id

    # Verify tools listing
    tools_list_response = client_with_models.tools.list(toolgroup_id=test_toolgroup_id)
    assert isinstance(tools_list_response, list)
    assert tools_list_response

    # Unregister the toolgroup
    client_with_models.toolgroups.unregister(toolgroup_id=test_toolgroup_id)

    # Verify unregistration
    unregistered_toolgroup = client_with_models.toolgroups.get(toolgroup_id=test_toolgroup_id)
    assert unregistered_toolgroup is None

    # Verify tools are also unregistered
    unregister_tools_list_response = client_with_models.tools.list(toolgroup_id=test_toolgroup_id)
    assert isinstance(unregister_tools_list_response, list)
    assert not unregister_tools_list_response
