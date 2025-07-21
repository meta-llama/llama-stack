# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import ssl
import time
from abc import ABC, abstractmethod
from asyncio import Lock
from urllib.parse import parse_qs, urlparse

import httpx
from jose import jwt
from pydantic import BaseModel, Field

from llama_stack.distribution.datatypes import (
    AuthenticationConfig,
    CustomAuthConfig,
    GitHubTokenAuthConfig,
    KubernetesAuthProviderConfig,
    OAuth2TokenAuthConfig,
    User,
)
from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="auth")


class AuthResponse(BaseModel):
    """The format of the authentication response from the auth endpoint."""

    principal: str
    # further attributes that may be used for access control decisions
    attributes: dict[str, list[str]] | None = None
    message: str | None = Field(
        default=None, description="Optional message providing additional context about the authentication result."
    )


class AuthRequestContext(BaseModel):
    path: str = Field(description="The path of the request being authenticated")

    headers: dict[str, str] = Field(description="HTTP headers from the original request (excluding Authorization)")

    params: dict[str, list[str]] = Field(default_factory=dict, description="Query parameters from the original request")


class AuthRequest(BaseModel):
    api_key: str = Field(description="The API key extracted from the Authorization header")

    request: AuthRequestContext = Field(description="Context information about the request being authenticated")


class AuthProvider(ABC):
    """Abstract base class for authentication providers."""

    @abstractmethod
    async def validate_token(self, token: str, scope: dict | None = None) -> User:
        """Validate a token and return access attributes."""
        pass

    @abstractmethod
    async def close(self):
        """Clean up any resources."""
        pass

    def get_auth_error_message(self, scope: dict | None = None) -> str:
        """Return provider-specific authentication error message."""
        return "Authentication required"


def get_attributes_from_claims(claims: dict[str, str], mapping: dict[str, str]) -> dict[str, list[str]]:
    attributes: dict[str, list[str]] = {}
    for claim_key, attribute_key in mapping.items():
        if claim_key not in claims:
            continue
        claim = claims[claim_key]
        if isinstance(claim, list):
            values = claim
        else:
            values = claim.split()

        if attribute_key in attributes:
            attributes[attribute_key].extend(values)
        else:
            attributes[attribute_key] = values
    return attributes


class OAuth2TokenAuthProvider(AuthProvider):
    """
    JWT token authentication provider that validates a JWT token and extracts access attributes.

    This should be the standard authentication provider for most use cases.
    """

    def __init__(self, config: OAuth2TokenAuthConfig):
        self.config = config
        self._jwks_at: float = 0.0
        self._jwks: dict[str, str] = {}
        self._jwks_lock = Lock()

    async def validate_token(self, token: str, scope: dict | None = None) -> User:
        if self.config.jwks:
            return await self.validate_jwt_token(token, scope)
        if self.config.introspection:
            return await self.introspect_token(token, scope)
        raise ValueError("One of jwks or introspection must be configured")

    async def validate_jwt_token(self, token: str, scope: dict | None = None) -> User:
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
                issuer=self.config.issuer,
            )
        except Exception as exc:
            raise ValueError("Invalid JWT token") from exc

        # There are other standard claims, the most relevant of which is `scope`.
        # We should incorporate these into the access attributes.
        principal = claims["sub"]
        access_attributes = get_attributes_from_claims(claims, self.config.claims_mapping)
        return User(
            principal=principal,
            attributes=access_attributes,
        )

    async def introspect_token(self, token: str, scope: dict | None = None) -> User:
        """Validate a token using token introspection as defined by RFC 7662."""
        form = {
            "token": token,
        }
        if self.config.introspection is None:
            raise ValueError("Introspection is not configured")

        if self.config.introspection.send_secret_in_body:
            form["client_id"] = self.config.introspection.client_id
            form["client_secret"] = self.config.introspection.client_secret
            auth = None
        else:
            auth = (self.config.introspection.client_id, self.config.introspection.client_secret)
        ssl_ctxt = None
        if self.config.tls_cafile:
            ssl_ctxt = ssl.create_default_context(cafile=self.config.tls_cafile.as_posix())
        try:
            async with httpx.AsyncClient(verify=ssl_ctxt) as client:
                response = await client.post(
                    self.config.introspection.url,
                    data=form,
                    auth=auth,
                    timeout=10.0,  # Add a reasonable timeout
                )
                if response.status_code != 200:
                    logger.warning(f"Token introspection failed with status code: {response.status_code}")
                    raise ValueError(f"Token introspection failed: {response.status_code}")

                fields = response.json()
                if not fields["active"]:
                    raise ValueError("Token not active")
                principal = fields["sub"] or fields["username"]
                access_attributes = get_attributes_from_claims(fields, self.config.claims_mapping)
                return User(
                    principal=principal,
                    attributes=access_attributes,
                )
        except httpx.TimeoutException:
            logger.warning("Token introspection request timed out")
            raise
        except ValueError:
            # Re-raise ValueError exceptions to preserve their message
            raise
        except Exception as e:
            logger.exception("Error during token introspection")
            raise ValueError("Token introspection error") from e

    async def close(self):
        pass

    def get_auth_error_message(self, scope: dict | None = None) -> str:
        """Return OAuth2-specific authentication error message."""
        if self.config.issuer:
            return f"Authentication required. Please provide a valid OAuth2 Bearer token from {self.config.issuer}"
        elif self.config.introspection:
            # Extract domain from introspection URL for a cleaner message
            domain = urlparse(self.config.introspection.url).netloc
            return f"Authentication required. Please provide a valid OAuth2 Bearer token validated by {domain}"
        else:
            return "Authentication required. Please provide a valid OAuth2 Bearer token in the Authorization header"

    async def _refresh_jwks(self) -> None:
        """
        Refresh the JWKS cache.

        This is a simple cache that expires after a certain amount of time (defined by `key_recheck_period`).
        If the cache is expired, we refresh the JWKS from the JWKS URI.

        Notes: for Kubernetes which doesn't fully implement the OIDC protocol:
            * It doesn't have user authentication flows
            * It doesn't have refresh tokens
        """
        async with self._jwks_lock:
            if self.config.jwks is None:
                raise ValueError("JWKS is not configured")
            if time.time() - self._jwks_at > self.config.jwks.key_recheck_period:
                headers = {}
                if self.config.jwks.token:
                    headers["Authorization"] = f"Bearer {self.config.jwks.token}"
                verify = self.config.tls_cafile.as_posix() if self.config.tls_cafile else self.config.verify_tls
                async with httpx.AsyncClient(verify=verify) as client:
                    res = await client.get(self.config.jwks.uri, timeout=5, headers=headers)
                    res.raise_for_status()
                    jwks_data = res.json()["keys"]
                    updated = {}
                    for k in jwks_data:
                        kid = k["kid"]
                        # Store the entire key object as it may be needed for different algorithms
                        updated[kid] = k
                    self._jwks = updated
                    self._jwks_at = time.time()


class CustomAuthProvider(AuthProvider):
    """Custom authentication provider that uses an external endpoint."""

    def __init__(self, config: CustomAuthConfig):
        self.config = config
        self._client = None

    async def validate_token(self, token: str, scope: dict | None = None) -> User:
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
                    return User(principal=auth_response.principal, attributes=auth_response.attributes)
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

    def get_auth_error_message(self, scope: dict | None = None) -> str:
        """Return custom auth provider-specific authentication error message."""
        domain = urlparse(self.config.endpoint).netloc
        if domain:
            return f"Authentication required. Please provide your API key as a Bearer token (validated by {domain})"
        else:
            return "Authentication required. Please provide your API key as a Bearer token in the Authorization header"


class GitHubTokenAuthProvider(AuthProvider):
    """
    GitHub token authentication provider that validates GitHub access tokens directly.

    This provider accepts GitHub personal access tokens or OAuth tokens and verifies
    them against the GitHub API to get user information.
    """

    def __init__(self, config: GitHubTokenAuthConfig):
        self.config = config

    async def validate_token(self, token: str, scope: dict | None = None) -> User:
        """Validate a GitHub token by calling the GitHub API.

        This validates tokens issued by GitHub (personal access tokens or OAuth tokens).
        """
        try:
            user_info = await _get_github_user_info(token, self.config.github_api_base_url)
        except httpx.HTTPStatusError as e:
            logger.warning(f"GitHub token validation failed: {e}")
            raise ValueError("GitHub token validation failed. Please check your token and try again.") from e

        principal = user_info["user"]["login"]

        github_data = {
            "login": user_info["user"]["login"],
            "id": str(user_info["user"]["id"]),
            "organizations": user_info.get("organizations", []),
        }

        access_attributes = get_attributes_from_claims(github_data, self.config.claims_mapping)

        return User(
            principal=principal,
            attributes=access_attributes,
        )

    async def close(self):
        """Clean up any resources."""
        pass

    def get_auth_error_message(self, scope: dict | None = None) -> str:
        """Return GitHub-specific authentication error message."""
        return "Authentication required. Please provide a valid GitHub access token (https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) in the Authorization header (Bearer <token>)"


async def _get_github_user_info(access_token: str, github_api_base_url: str) -> dict:
    """Fetch user info and organizations from GitHub API."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "llama-stack",
    }

    async with httpx.AsyncClient() as client:
        user_response = await client.get(f"{github_api_base_url}/user", headers=headers, timeout=10.0)
        user_response.raise_for_status()
        user_data = user_response.json()

        return {
            "user": user_data,
        }


class KubernetesAuthProvider(AuthProvider):
    """
    Kubernetes authentication provider that validates tokens using the Kubernetes SelfSubjectReview API.
    This provider integrates with Kubernetes API server by using the
    /apis/authentication.k8s.io/v1/selfsubjectreviews endpoint to validate tokens and extract user information.
    """

    def __init__(self, config: KubernetesAuthProviderConfig):
        self.config = config

    async def validate_token(self, token: str, scope: dict | None = None) -> User:
        """Validate a token using Kubernetes SelfSubjectReview API endpoint."""

        # Configure SSL context
        ssl_ctxt = None
        if self.config.tls_cafile:
            ssl_ctxt = ssl.create_default_context(cafile=self.config.tls_cafile.as_posix())
        elif not self.config.verify_tls:
            ssl_ctxt = ssl.create_default_context()
            ssl_ctxt.check_hostname = False
            ssl_ctxt.verify_mode = ssl.CERT_NONE

        # Build the Kubernetes SelfSubjectReview API endpoint URL
        review_api_url = f"{self.config.api_server_url.rstrip('/')}/apis/authentication.k8s.io/v1/selfsubjectreviews"

        # Create SelfSubjectReview request body
        review_request = {"apiVersion": "authentication.k8s.io/v1", "kind": "SelfSubjectReview"}

        try:
            async with httpx.AsyncClient(verify=ssl_ctxt if ssl_ctxt else self.config.verify_tls) as client:
                response = await client.post(
                    review_api_url,
                    json=review_request,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    timeout=10.0,
                )

                if response.status_code == 401:
                    raise ValueError("Invalid token")
                elif response.status_code != 201:
                    logger.warning(f"Kubernetes SelfSubjectReview API failed with status code: {response.status_code}")
                    raise ValueError(f"Token validation failed: {response.status_code}")

                review_response = response.json()

                # Extract user information from SelfSubjectReview response
                status = review_response.get("status", {})
                if not status:
                    raise ValueError("No status found in SelfSubjectReview response")

                user_info = status.get("userInfo", {})
                if not user_info:
                    raise ValueError("No userInfo found in SelfSubjectReview response")

                username = user_info.get("username")
                if not username:
                    raise ValueError("No username found in SelfSubjectReview response")

                # Build user attributes from Kubernetes user info
                user_attributes = get_attributes_from_claims(user_info, self.config.claims_mapping)

                return User(
                    principal=username,
                    attributes=user_attributes,
                )

        except httpx.TimeoutException:
            logger.warning("Kubernetes SelfSubjectReview API request timed out")
            raise ValueError("Token validation timeout") from None
        except ValueError:
            # Re-raise ValueError exceptions to preserve their message
            raise
        except Exception as e:
            logger.warning(f"Error during token validation: {str(e)}")
            raise ValueError(f"Token validation error: {str(e)}") from e

    async def close(self):
        """Close any resources."""
        pass


def create_auth_provider(config: AuthenticationConfig) -> AuthProvider:
    """Factory function to create the appropriate auth provider."""
    provider_config = config.provider_config

    if isinstance(provider_config, CustomAuthConfig):
        return CustomAuthProvider(provider_config)
    elif isinstance(provider_config, OAuth2TokenAuthConfig):
        return OAuth2TokenAuthProvider(provider_config)
    elif isinstance(provider_config, GitHubTokenAuthConfig):
        return GitHubTokenAuthProvider(provider_config)
    elif isinstance(provider_config, KubernetesAuthProviderConfig):
        return KubernetesAuthProvider(provider_config)
    else:
        raise ValueError(f"Unknown authentication provider config type: {type(provider_config)}")
