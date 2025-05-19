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
from pydantic import BaseModel, Field, field_validator

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
    OAUTH2_TOKEN = "oauth2_token"


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


def get_attributes_from_claims(claims: dict[str, str], mapping: dict[str, str]) -> AccessAttributes:
    attributes = AccessAttributes()
    for claim_key, attribute_key in mapping.items():
        if claim_key not in claims or not hasattr(attributes, attribute_key):
            continue
        claim = claims[claim_key]
        if isinstance(claim, list):
            values = claim
        else:
            values = claim.split()

        current = getattr(attributes, attribute_key)
        if current:
            current.extend(values)
        else:
            setattr(attributes, attribute_key, values)
    return attributes


class OAuth2TokenAuthProviderConfig(BaseModel):
    # The JWKS URI for collecting public keys
    jwks_uri: str
    cache_ttl: int = 3600
    audience: str = "llama-stack"
    claims_mapping: dict[str, str] = Field(
        default_factory=lambda: {
            "sub": "roles",
            "username": "roles",
            "groups": "teams",
            "team": "teams",
            "project": "projects",
            "tenant": "namespaces",
            "namespace": "namespaces",
        },
    )

    @classmethod
    @field_validator("claims_mapping")
    def validate_claims_mapping(cls, v):
        for key, value in v.items():
            if not value:
                raise ValueError(f"claims_mapping value cannot be empty: {key}")
            if value not in AccessAttributes.model_fields:
                raise ValueError(f"claims_mapping value is not a valid attribute: {value}")
        return v


class OAuth2TokenAuthProvider(AuthProvider):
    """
    JWT token authentication provider that validates a JWT token and extracts access attributes.

    This should be the standard authentication provider for most use cases.
    """

    def __init__(self, config: OAuth2TokenAuthProviderConfig):
        self.config = config
        self._jwks_at: float = 0.0
        self._jwks: dict[str, str] = {}

    async def validate_token(self, token: str, scope: dict | None = None) -> TokenValidationResult:
        """Validate a token using the JWT token."""
        await self._refresh_jwks()

        try:
            header = jwt.get_unverified_header(token)
            kid = header["kid"]
            if kid not in self._jwks:
                raise ValueError(f"Unknown key ID: {kid}")
            key_data = self._jwks[kid]
            algorithm = header.get("alg", "RS256")
            claims = jwt.decode(
                token,
                key_data,
                algorithms=[algorithm],
                audience=self.config.audience,
                options={"verify_exp": True},
            )
        except Exception as exc:
            raise ValueError(f"Invalid JWT token: {token}") from exc

        # There are other standard claims, the most relevant of which is `scope`.
        # We should incorporate these into the access attributes.
        principal = claims["sub"]
        access_attributes = get_attributes_from_claims(claims, self.config.claims_mapping)
        return TokenValidationResult(
            principal=principal,
            access_attributes=access_attributes,
        )

    async def close(self):
        """Close the HTTP client."""

    async def _refresh_jwks(self) -> None:
        if time.time() - self._jwks_at > self.config.cache_ttl:
            async with httpx.AsyncClient() as client:
                res = await client.get(self.config.jwks_uri, timeout=5)
                res.raise_for_status()
                jwks_data = res.json()["keys"]
                self._jwks = {}
                for k in jwks_data:
                    kid = k["kid"]
                    # Store the entire key object as it may be needed for different algorithms
                    self._jwks[kid] = k
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
    elif provider_type == "oauth2_token":
        return OAuth2TokenAuthProvider(OAuth2TokenAuthProviderConfig.model_validate(config.config))
    else:
        supported_providers = ", ".join([t.value for t in AuthProviderType])
        raise ValueError(f"Unsupported auth provider type: {provider_type}. Supported types are: {supported_providers}")
