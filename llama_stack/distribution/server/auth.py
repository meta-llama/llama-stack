# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from urllib.parse import parse_qs

import httpx

from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="auth")


class AuthenticationMiddleware:
    def __init__(self, app, auth_endpoint):
        self.app = app
        self.auth_endpoint = auth_endpoint

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            auth_header = headers.get(b"authorization", b"").decode()

            if not auth_header or not auth_header.startswith("Bearer "):
                return await self._send_auth_error(send, "Missing or invalid Authorization header")

            api_key = auth_header.split("Bearer ", 1)[1]

            path = scope.get("path", "")
            request_headers = {k.decode(): v.decode() for k, v in headers.items()}

            query_string = scope.get("query_string", b"").decode()
            params = parse_qs(query_string)

            auth_data = {
                "api_key": api_key,
                "request": {
                    "path": path,
                    "headers": request_headers,
                    "params": params,
                },
            }

            # Validate with authentication endpoint
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(self.auth_endpoint, json=auth_data)
                    if response.status_code != 200:
                        logger.warning(f"Authentication failed: {response.status_code}")
                        return await self._send_auth_error(send, "Authentication failed")
            except Exception:
                logger.exception("Error during authentication")
                return await self._send_auth_error(send, "Authentication service error")

        return await self.app(scope, receive, send)

    async def _send_auth_error(self, send, message):
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [[b"content-type", b"application/json"]],
            }
        )
        error_msg = json.dumps({"error": {"message": message}}).encode()
        await send({"type": "http.response.body", "body": error_msg})
