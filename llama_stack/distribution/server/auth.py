# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from typing import Dict, List, Optional
from urllib.parse import parse_qs

import httpx
from pydantic import BaseModel, Field

from llama_stack.distribution.datatypes import AccessAttributes
from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="auth")


class AuthRequestContext(BaseModel):
    path: str = Field(description="The path of the request being authenticated")

    headers: Dict[str, str] = Field(description="HTTP headers from the original request (excluding Authorization)")

    params: Dict[str, List[str]] = Field(
        description="Query parameters from the original request, parsed as dictionary of lists"
    )


class AuthRequest(BaseModel):
    api_key: str = Field(description="The API key extracted from the Authorization header")

    request: AuthRequestContext = Field(description="Context information about the request being authenticated")


class AuthResponse(BaseModel):
    """The format of the authentication response from the auth endpoint."""

    access_attributes: Optional[AccessAttributes] = Field(
        default=None,
        description="""
        Structured user attributes for attribute-based access control.

        These attributes determine which resources the user can access.
        The model provides standard categories like "roles", "teams", "projects", and "namespaces".
        Each attribute category contains a list of values that the user has for that category.
        During access control checks, these values are compared against resource requirements.

        Example with standard categories:
        ```json
        {
            "roles": ["admin", "data-scientist"],
            "teams": ["ml-team"],
            "projects": ["llama-3"],
            "namespaces": ["research"]
        }
        ```
        """,
    )

    message: Optional[str] = Field(
        default=None, description="Optional message providing additional context about the authentication result."
    )


class AuthenticationMiddleware:
    """Middleware that authenticates requests using an external auth endpoint.

    This middleware:
    1. Extracts the Bearer token from the Authorization header
    2. Sends it to the configured auth endpoint along with request details
    3. Validates the response and extracts user attributes
    4. Makes these attributes available to the route handlers for access control

    Authentication Request Format:
    ```json
    {
        "api_key": "the-api-key-extracted-from-auth-header",
        "request": {
            "path": "/models/list",
            "headers": {
                "content-type": "application/json",
                "user-agent": "..."
                // All headers except Authorization
            },
            "params": {
                "limit": ["100"],
                "offset": ["0"]
                // Query parameters as key -> list of values
            }
        }
    }
    ```

    Expected Auth Endpoint Response Format:
    ```json
    {
        "access_attributes": {    // Structured attribute format
            "roles": ["admin", "user"],
            "teams": ["ml-team", "nlp-team"],
            "projects": ["llama-3", "project-x"],
            "namespaces": ["research"]
        },
        "message": "Optional message about auth result"
    }
    ```

    Attribute-Based Access Control:
    The attributes returned by the auth endpoint are used to determine which
    resources the user can access. Resources can specify required attributes
    using the access_attributes field. For a user to access a resource:

    1. All attribute categories specified in the resource must be present in the user's attributes
    2. For each category, the user must have at least one matching value

    If the auth endpoint doesn't return any attributes, the user will only be able to
    access resources that don't have access_attributes defined.
    """

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

            # Remove sensitive headers
            if "authorization" in request_headers:
                del request_headers["authorization"]

            query_string = scope.get("query_string", b"").decode()
            params = parse_qs(query_string)

            # Build the auth request model
            auth_request = AuthRequest(
                api_key=api_key,
                request=AuthRequestContext(
                    path=path,
                    headers=request_headers,
                    params=params,
                ),
            )

            # Validate with authentication endpoint
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self.auth_endpoint,
                        json=auth_request.model_dump(),
                        timeout=10.0,  # Add a reasonable timeout
                    )
                    if response.status_code != 200:
                        logger.warning(f"Authentication failed: {response.status_code}")
                        return await self._send_auth_error(send, "Authentication failed")

                    # Parse and validate the auth response
                    try:
                        response_data = response.json()
                        auth_response = AuthResponse(**response_data)

                        # Store attributes in request scope for access control
                        if auth_response.access_attributes:
                            user_attributes = auth_response.access_attributes.model_dump(exclude_none=True)
                        else:
                            logger.warning("No access attributes, setting namespace to api_key by default")
                            user_attributes = {
                                "namespaces": [api_key],
                            }

                        scope["user_attributes"] = user_attributes
                        logger.debug(f"Authentication successful: {len(user_attributes)} attributes")
                    except Exception:
                        logger.exception("Error parsing authentication response")
                        return await self._send_auth_error(send, "Invalid authentication response format")
            except httpx.TimeoutException:
                logger.exception("Authentication request timed out")
                return await self._send_auth_error(send, "Authentication service timeout")
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
