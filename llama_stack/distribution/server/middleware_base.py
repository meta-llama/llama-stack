# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from abc import ABC, abstractmethod


class BaseServerMiddleware(ABC):
    """Base class for server middleware that need route matching capabilities"""

    def __init__(self, app, impls):
        self.app = app
        self.impls = impls
        # FastAPI built-in paths that should bypass custom routing
        self.fastapi_paths = ("/docs", "/redoc", "/openapi.json", "/favicon.ico", "/static")

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "lifespan":
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        method = scope.get("method", "GET")

        # Check if the path is a FastAPI built-in path
        if path.startswith(self.fastapi_paths):
            return await self.app(scope, receive, send)

        # Initialize route implementations if needed
        if not hasattr(self, "route_impls"):
            from llama_stack.distribution.server.routes import initialize_route_impls

            self.route_impls = initialize_route_impls(self.impls)

        # Find the matching route and its implementation
        try:
            from llama_stack.distribution.server.routes import find_matching_route

            route, impl, webmethod = find_matching_route(method, path, self.route_impls)
        except ValueError:
            # If no matching endpoint is found, pass through to FastAPI
            return await self.app(scope, receive, send)

        # Call the middleware-specific processing
        return await self.process_request(scope, receive, send, route, impl, webmethod)

    @abstractmethod
    async def process_request(self, scope, receive, send, route, impl, webmethod):
        """Process the request with middleware-specific logic"""
        pass
