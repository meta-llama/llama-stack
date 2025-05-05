# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from abc import ABC, abstractmethod
from enum import Enum
from urllib.parse import parse_qs

import httpx
from pydantic import BaseModel, Field

from llama_stack.distribution.datatypes import AccessAttributes
from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="auth")


class AuthResponse(BaseModel):
    """The format of the authentication response from the auth endpoint."""

    access_attributes: AccessAttributes | None = Field(
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

    message: str | None = Field(
        default=None, description="Optional message providing additional context about the authentication result."
    )


class AuthRequestContext(BaseModel):
    path: str = Field(description="The path of the request being authenticated")

    headers: dict[str, str] = Field(description="HTTP headers from the original request (excluding Authorization)")

    params: dict[str, list[str]] = Field(
        description="Query parameters from the original request, parsed as dictionary of lists"
    )


class AuthRequest(BaseModel):
    api_key: str = Field(description="The API key extracted from the Authorization header")

    request: AuthRequestContext = Field(description="Context information about the request being authenticated")


class AuthProviderType(str, Enum):
    """Supported authentication provider types."""

    KUBERNETES = "kubernetes"
    CUSTOM = "custom"


class AuthProviderConfig(BaseModel):
    """Base configuration for authentication providers."""

    provider_type: AuthProviderType = Field(..., description="Type of authentication provider")
    config: dict[str, str] = Field(..., description="Provider-specific configuration")


class AuthProvider(ABC):
    """Abstract base class for authentication providers."""

    @abstractmethod
    async def validate_token(self, token: str, scope: dict | None = None) -> AccessAttributes | None:
        """Validate a token and return access attributes."""
        pass

    @abstractmethod
    async def close(self):
        """Clean up any resources."""
        pass


class KubernetesAuthProvider(AuthProvider):
    """Kubernetes authentication provider that validates tokens against the Kubernetes API server."""

    def __init__(self, config: dict[str, str]):
        self.api_server_url = config["api_server_url"]
        self.ca_cert_path = config.get("ca_cert_path")
        self._client = None

    async def _get_client(self):
        """Get or create a Kubernetes client."""
        if self._client is None:
            # kubernetes-client has not async support, see:
            # https://github.com/kubernetes-client/python/issues/323
            from kubernetes import client
            from kubernetes.client import ApiClient

            # Configure the client
            configuration = client.Configuration()
            configuration.host = self.api_server_url
            if self.ca_cert_path:
                configuration.ssl_ca_cert = self.ca_cert_path
            configuration.verify_ssl = bool(self.ca_cert_path)

            # Create API client
            self._client = ApiClient(configuration)
        return self._client

    async def validate_token(self, token: str, scope: dict | None = None) -> AccessAttributes | None:
        """Validate a Kubernetes token and return access attributes."""
        try:
            client = await self._get_client()

            # Set the token in the client
            client.set_default_header("Authorization", f"Bearer {token}")

            # Make a request to validate the token
            # We use the /api endpoint which requires authentication
            from kubernetes.client import CoreV1Api

            api = CoreV1Api(client)
            api.get_api_resources(_request_timeout=3.0)  # Set timeout for this specific request

            # If we get here, the token is valid
            # Extract user info from the token claims
            import base64

            # Decode the token (without verification since we've already validated it)
            token_parts = token.split(".")
            payload = json.loads(base64.b64decode(token_parts[1] + "=" * (-len(token_parts[1]) % 4)))

            # Extract user information from the token
            username = payload.get("sub", "")
            groups = payload.get("groups", [])

            return AccessAttributes(
                roles=[username],  # Use username as a role
                teams=groups,  # Use Kubernetes groups as teams
            )

        except Exception as e:
            logger.exception("Failed to validate Kubernetes token")
            raise ValueError("Invalid or expired token") from e

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None


class CustomAuthProvider(AuthProvider):
    """Custom authentication provider that uses an external endpoint."""

    def __init__(self, config: dict[str, str]):
        self.endpoint = config["endpoint"]
        self._client = None

    async def validate_token(self, token: str, scope: dict | None = None) -> AccessAttributes | None:
        """Validate a token using the custom authentication endpoint."""
        if not self.endpoint:
            raise ValueError("Authentication endpoint not configured")

        if scope is None:
            scope = {}

        headers = dict(scope.get("headers", []))
        path = scope.get("path", "")
        request_headers = {k.decode(): v.decode() for k, v in headers.items()}

        # Remove sensitive headers
        if "authorization" in request_headers:
            del request_headers["authorization"]

        query_string = scope.get("query_string", b"").decode()
        params = parse_qs(query_string)

        # Build the auth request model
        auth_request = AuthRequest(
            api_key=token,
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
                    self.endpoint,
                    json=auth_request.model_dump(),
                    timeout=10.0,  # Add a reasonable timeout
                )
                if response.status_code != 200:
                    logger.warning(f"Authentication failed with status code: {response.status_code}")
                    raise ValueError(f"Authentication failed: {response.status_code}")

                # Parse and validate the auth response
                try:
                    response_data = response.json()
                    auth_response = AuthResponse(**response_data)

                    # Store attributes in request scope for access control
                    if auth_response.access_attributes:
                        return auth_response.access_attributes
                    else:
                        logger.warning("No access attributes, setting namespace to api_key by default")
                        user_attributes = {
                            "namespaces": [token],
                        }

                    scope["user_attributes"] = user_attributes
                    logger.debug(f"Authentication successful: {len(user_attributes)} attributes")
                    return auth_response.access_attributes
                except Exception as e:
                    logger.exception("Error parsing authentication response")
                    raise ValueError("Invalid authentication response format") from e

        except httpx.TimeoutException:
            logger.exception("Authentication request timed out")
            raise
        except ValueError:
            # Re-raise ValueError exceptions to preserve their message
            raise
        except Exception as e:
            logger.exception("Error during authentication")
            raise ValueError("Authentication service error") from e

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


def create_auth_provider(config: AuthProviderConfig) -> AuthProvider:
    """Factory function to create the appropriate auth provider."""
    provider_type = config.provider_type.lower()

    if provider_type == "kubernetes":
        return KubernetesAuthProvider(config.config)
    elif provider_type == "custom":
        return CustomAuthProvider(config.config)
    else:
        supported_providers = ", ".join([t.value for t in AuthProviderType])
        raise ValueError(f"Unsupported auth provider type: {provider_type}. Supported types are: {supported_providers}")
