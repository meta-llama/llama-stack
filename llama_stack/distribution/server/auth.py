# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

import httpx

from llama_stack.distribution.datatypes import AuthenticationConfig
from llama_stack.distribution.server.auth_providers import create_auth_provider
from llama_stack.distribution.server.oauth2_scopes import get_required_scopes_for_api
from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="auth")


def extract_api_from_path(path: str) -> tuple[str, str]:
    """Extract API name and method from request path for scope validation"""
    # Remove leading/trailing slashes and split
    path = path.strip("/")
    parts = path.split("/")

    # Handle common API path patterns
    if len(parts) >= 2 and parts[0] == "v1":
        api_name = parts[1]
        # Handle nested paths like /v1/models/{id} or /v1/inference/chat-completion
        if api_name in ["inference", "models", "agents", "tools", "vector_dbs", "safety", "eval", "scoring"]:
            return api_name, "POST"  # Default to POST for scope checking
        elif api_name == "openai":
            # Handle OpenAI compatibility endpoints like /v1/openai/v1/chat/completions
            if len(parts) >= 4:
                return "inference", "POST"  # OpenAI endpoints are typically inference

    # Fallback - try to extract from first path component
    if parts:
        return parts[0], "POST"

    return "unknown", "GET"


class AuthenticationMiddleware:
    """Middleware that authenticates requests using configured authentication provider.

    This middleware:
    1. Extracts the Bearer token from the Authorization header
    2. Uses the configured auth provider to validate the token
    3. Extracts user attributes from the provider's response
    4. Makes these attributes available to the route handlers for access control

    The middleware supports multiple authentication providers through the AuthProvider interface:
    - Kubernetes: Validates tokens against the Kubernetes API server
    - Custom: Validates tokens against a custom endpoint

    Authentication Request Format for Custom Auth Provider:
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

    Token Validation:
    Each provider implements its own token validation logic:
    - Kubernetes: Uses TokenReview API to validate service account tokens
    - Custom: Sends token to custom endpoint for validation

    Attribute-Based Access Control:
    The attributes returned by the auth provider are used to determine which
    resources the user can access. Resources can specify required attributes
    using the access_attributes field. For a user to access a resource:

    1. All attribute categories specified in the resource must be present in the user's attributes
    2. For each category, the user must have at least one matching value

    If the auth provider doesn't return any attributes, the user will only be able to
    access resources that don't have access_attributes defined.
    """

    def __init__(self, app, auth_config: AuthenticationConfig):
        self.app = app
        self.auth_provider = create_auth_provider(auth_config)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            auth_header = headers.get(b"authorization", b"").decode()

            if not auth_header:
                error_msg = self.auth_provider.get_auth_error_message(scope)
                return await self._send_auth_error(send, error_msg)

            if not auth_header.startswith("Bearer "):
                return await self._send_auth_error(send, "Invalid Authorization header format")

            token = auth_header.split("Bearer ", 1)[1]

            # Validate token and get access attributes
            try:
                validation_result = await self.auth_provider.validate_token(token, scope)
            except httpx.TimeoutException:
                logger.exception("Authentication request timed out")
                return await self._send_auth_error(send, "Authentication service timeout")
            except ValueError as e:
                logger.exception("Error during authentication")
                return await self._send_auth_error(send, str(e))
            except Exception:
                logger.exception("Error during authentication")
                return await self._send_auth_error(send, "Authentication service error")

            # Validate OAuth2 scopes for the requested API endpoint
            path = scope.get("path", "")
            method = scope.get("method", "GET")
            api_name, _ = extract_api_from_path(path)

            # Get required scopes for this API endpoint
            required_scopes = get_required_scopes_for_api(api_name, method)

            # Check if user has any of the required scopes
            user_scopes = set()
            if validation_result.attributes and "scopes" in validation_result.attributes:
                user_scopes = set(validation_result.attributes["scopes"])

            # Verify user has at least one required scope
            if not user_scopes.intersection(required_scopes):
                logger.warning(
                    f"Access denied for {validation_result.principal} to {api_name} API. "
                    f"Required scopes: {required_scopes}, User scopes: {user_scopes}"
                )
                return await self._send_auth_error(
                    send, f"Insufficient OAuth2 scopes for {api_name} API. Required: {', '.join(required_scopes)}"
                )

            logger.debug(
                f"OAuth2 scope validation passed for {validation_result.principal} "
                f"on {api_name} API with scopes: {user_scopes.intersection(required_scopes)}"
            )

            # Store the client ID in the request scope so that downstream middleware (like QuotaMiddleware)
            # can identify the requester and enforce per-client rate limits.
            scope["authenticated_client_id"] = token

            # Store attributes in request scope
            scope["principal"] = validation_result.principal
            if validation_result.attributes:
                scope["user_attributes"] = validation_result.attributes
            attr_count = len(validation_result.attributes) if validation_result.attributes else 0
            logger.debug(f"Authentication successful: {validation_result.principal} with {attr_count} attributes")

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
