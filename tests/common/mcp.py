# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# we want the mcp server to be authenticated OR not, depends
from contextlib import contextmanager

# Unfortunately the toolgroup id must be tied to the tool names because the registry
# indexes on both toolgroups and tools independently (and not jointly). That really
# needs to be fixed.
MCP_TOOLGROUP_ID = "mcp::localmcp"


@contextmanager
def make_mcp_server(required_auth_token: str | None = None):
    import threading
    import time

    import httpx
    import uvicorn
    from mcp import types
    from mcp.server.fastmcp import Context, FastMCP
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.responses import Response
    from starlette.routing import Mount, Route

    server = FastMCP("FastMCP Test Server", log_level="WARNING")

    @server.tool()
    async def greet_everyone(
        url: str, ctx: Context
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        return [types.TextContent(type="text", text="Hello, world!")]

    @server.tool()
    async def get_boiling_point(liquid_name: str, celcius: bool = True) -> int:
        """
        Returns the boiling point of a liquid in Celcius or Fahrenheit.

        :param liquid_name: The name of the liquid
        :param celcius: Whether to return the boiling point in Celcius
        :return: The boiling point of the liquid in Celcius or Fahrenheit
        """
        if liquid_name.lower() == "polyjuice":
            if celcius:
                return -100
            else:
                return -212
        else:
            return -1

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        from starlette.exceptions import HTTPException

        auth_header = request.headers.get("Authorization")
        auth_token = None
        if auth_header and auth_header.startswith("Bearer "):
            auth_token = auth_header.split(" ")[1]

        if required_auth_token and auth_token != required_auth_token:
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
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    port = get_open_port()

    # make uvicorn logs be less verbose
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
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

    server_url = f"http://localhost:{port}/sse"
    while time.time() - start_time < timeout:
        try:
            response = httpx.get(server_url)
            if response.status_code in [200, 401]:
                break
        except httpx.RequestError:
            pass
        time.sleep(0.1)

    try:
        yield {"server_url": server_url}
    finally:
        print("Telling SSE server to exit")
        server_instance.should_exit = True
        time.sleep(0.5)

        # Force shutdown if still running
        if server_thread.is_alive():
            try:
                if hasattr(server_instance, "servers") and server_instance.servers:
                    for srv in server_instance.servers:
                        srv.close()

                # Wait for graceful shutdown
                server_thread.join(timeout=3)
                if server_thread.is_alive():
                    print("Warning: Server thread still alive after shutdown attempt")
            except Exception as e:
                print(f"Error during server shutdown: {e}")

        # CRITICAL: Reset SSE global state to prevent event loop contamination
        # Reset the SSE AppStatus singleton that stores anyio.Event objects
        from sse_starlette.sse import AppStatus

        AppStatus.should_exit = False
        AppStatus.should_exit_event = None
        print("SSE server exited")
