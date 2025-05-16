# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import time
from abc import ABC, abstractmethod
from enum import Enum
from urllib.parse import parse_qs

import httpx
from jose import jwt
from pydantic import BaseModel, Field

from llama_stack.distribution.datatypes import AccessAttributes
from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="auth")


class TokenValidationResult(BaseModel):
    principal: str | None = Field(
        default=None,
        description="The principal (username or persistent identifier) of the authenticated user",
    )
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


class AuthResponse(TokenValidationResult):
    """The format of the authentication response from the auth endpoint."""

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
    async def validate_token(self, token: str, scope: dict | None = None) -> TokenValidationResult:
        """Validate a token and return access attributes."""
        pass

    @abstractmethod
    async def close(self):
        """Clean up any resources."""
        pass


class KubernetesAuthProviderConfig(BaseModel):
    api_server_url: str
    ca_cert_path: str | None = None


class KubernetesAuthProvider(AuthProvider):
    """Kubernetes authentication provider that validates tokens against the Kubernetes API server."""

    def __init__(self, config: KubernetesAuthProviderConfig):
        self.config = config
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
            configuration.host = self.config.api_server_url
            if self.config.ca_cert_path:
                configuration.ssl_ca_cert = self.config.ca_cert_path
            configuration.verify_ssl = bool(self.config.ca_cert_path)

            # Create API client
            self._client = ApiClient(configuration)
        return self._client

    async def validate_token(self, token: str, scope: dict | None = None) -> TokenValidationResult:
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

            return TokenValidationResult(
                principal=username,
                access_attributes=AccessAttributes(
                    roles=[username],  # Use username as a role
                    teams=groups,  # Use Kubernetes groups as teams
                ),
            )

        except Exception as e:
            logger.exception("Failed to validate Kubernetes token")
            raise ValueError("Invalid or expired token") from e

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None


JWT_AUDIENCE = "llama-stack"


class JWKSAuthProviderConfig(BaseModel):
    """Configuration for JWT token authentication provider."""

    # The JWKS URI for collecting public keys
    jwks_uri: str
    algorithm: str = "RS256"
    cache_ttl: int = 3600


class JWKSAuthProvider(AuthProvider):
    """JWT token authentication provider that validates tokens against the JWT token."""

    def __init__(self, config: JWKSAuthProviderConfig):
        self.config = config
        self._jwks_at: float = 0.0
        self._jwks: dict[str, str] = {}

    async def validate_token(self, token: str, scope: dict | None = None) -> TokenValidationResult:
        """Validate a token using the JWT token."""
        await self._refresh_jwks()

        try:
            kid = jwt.get_unverified_header(token)["kid"]
            key = self._jwks[kid]  # raises if unknown
            claims = jwt.decode(
                token,
                key,
                algorithms=[self.config.algorithm],
                audience=JWT_AUDIENCE,
                options={"verify_exp": True},
            )
        except Exception as exc:
            raise ValueError(f"invalid token: {token}") from exc

        principal = f"{claims['iss']}:{claims['sub']}"

        teams = claims.get("teams", [])
        if not teams:
            if team := claims.get("team", claims.get("team_id")):
                teams = [team]
        projects = claims.get("projects", [])
        if not projects:
            if project := claims.get("project", claims.get("project_id")):
                projects = [project]
        namespaces = claims.get("namespaces", [])
        if not namespaces:
            if namespace := claims.get("namespace", claims.get("tenant")):
                namespaces = [namespace]

        return TokenValidationResult(
            principal=principal,
            access_attributes=AccessAttributes(
                roles=claims.get("groups", claims.get("roles", [])),  # Okta / Auth0
                teams=teams,
                projects=projects,
                namespaces=namespaces,
            ),
        )

    async def close(self):
        """Close the HTTP client."""

    async def _refresh_jwks(self) -> None:
        if time.time() - self._jwks_at > self.config.cache_ttl:
            with httpx.AsyncClient() as client:
                res = await client.get(self.config.jwks_uri, timeout=5)
                res.raise_for_status()
                self._jwks = {k["kid"]: k for k in res.json()["keys"]}
                self._jwks_at = time.time()


class CustomAuthProviderConfig(BaseModel):
    endpoint: str


class CustomAuthProvider(AuthProvider):
    """Custom authentication provider that uses an external endpoint."""

    def __init__(self, config: CustomAuthProviderConfig):
        self.config = config
        self._client = None

    async def validate_token(self, token: str, scope: dict | None = None) -> TokenValidationResult:
        """Validate a token using the custom authentication endpoint."""
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
                    self.config.endpoint,
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
                    return auth_response
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
        return KubernetesAuthProvider(KubernetesAuthProviderConfig.model_validate(config.config))
    elif provider_type == "custom":
        return CustomAuthProvider(CustomAuthProviderConfig.model_validate(config.config))
    elif provider_type == "jwks":
        return JWKSAuthProvider(JWKSAuthProviderConfig.model_validate(config.config))
    else:
        supported_providers = ", ".join([t.value for t in AuthProviderType])
        raise ValueError(f"Unsupported auth provider type: {provider_type}. Supported types are: {supported_providers}")
