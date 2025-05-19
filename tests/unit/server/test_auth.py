# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llama_stack.distribution.datatypes import AccessAttributes
from llama_stack.distribution.server.auth import AuthenticationMiddleware
from llama_stack.distribution.server.auth_providers import (
    AuthProviderConfig,
    AuthProviderType,
    TokenValidationResult,
    get_attributes_from_claims,
)


class MockResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception(f"HTTP error: {self.status_code}")


@pytest.fixture
def mock_auth_endpoint():
    return "http://mock-auth-service/validate"


@pytest.fixture
def valid_api_key():
    return "valid_api_key_12345"


@pytest.fixture
def invalid_api_key():
    return "invalid_api_key_67890"


@pytest.fixture
def valid_token():
    return "valid.jwt.token"


@pytest.fixture
def invalid_token():
    return "invalid.jwt.token"


@pytest.fixture
def http_app(mock_auth_endpoint):
    app = FastAPI()
    auth_config = AuthProviderConfig(
        provider_type=AuthProviderType.CUSTOM,
        config={"endpoint": mock_auth_endpoint},
    )
    app.add_middleware(AuthenticationMiddleware, auth_config=auth_config)

    @app.get("/test")
    def test_endpoint():
        return {"message": "Authentication successful"}

    return app


@pytest.fixture
def k8s_app():
    app = FastAPI()
    auth_config = AuthProviderConfig(
        provider_type=AuthProviderType.KUBERNETES,
        config={"api_server_url": "https://kubernetes.default.svc"},
    )
    app.add_middleware(AuthenticationMiddleware, auth_config=auth_config)

    @app.get("/test")
    def test_endpoint():
        return {"message": "Authentication successful"}

    return app


@pytest.fixture
def http_client(http_app):
    return TestClient(http_app)


@pytest.fixture
def k8s_client(k8s_app):
    return TestClient(k8s_app)


@pytest.fixture
def mock_scope():
    return {
        "type": "http",
        "path": "/models/list",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer test.jwt.token"),
            (b"user-agent", b"test-user-agent"),
        ],
        "query_string": b"limit=100&offset=0",
    }


@pytest.fixture
def mock_http_middleware(mock_auth_endpoint):
    mock_app = AsyncMock()
    auth_config = AuthProviderConfig(
        provider_type=AuthProviderType.CUSTOM,
        config={"endpoint": mock_auth_endpoint},
    )
    return AuthenticationMiddleware(mock_app, auth_config), mock_app


@pytest.fixture
def mock_k8s_middleware():
    mock_app = AsyncMock()
    auth_config = AuthProviderConfig(
        provider_type=AuthProviderType.KUBERNETES,
        config={"api_server_url": "https://kubernetes.default.svc"},
    )
    return AuthenticationMiddleware(mock_app, auth_config), mock_app


async def mock_post_success(*args, **kwargs):
    return MockResponse(
        200,
        {
            "message": "Authentication successful",
            "principal": "test-principal",
            "access_attributes": {
                "roles": ["admin", "user"],
                "teams": ["ml-team", "nlp-team"],
                "projects": ["llama-3", "project-x"],
                "namespaces": ["research", "production"],
            },
        },
    )


async def mock_post_failure(*args, **kwargs):
    return MockResponse(401, {"message": "Authentication failed"})


async def mock_post_exception(*args, **kwargs):
    raise Exception("Connection error")


# HTTP Endpoint Tests
def test_missing_auth_header(http_client):
    response = http_client.get("/test")
    assert response.status_code == 401
    assert "Missing or invalid Authorization header" in response.json()["error"]["message"]


def test_invalid_auth_header_format(http_client):
    response = http_client.get("/test", headers={"Authorization": "InvalidFormat token123"})
    assert response.status_code == 401
    assert "Missing or invalid Authorization header" in response.json()["error"]["message"]


@patch("httpx.AsyncClient.post", new=mock_post_success)
def test_valid_http_authentication(http_client, valid_api_key):
    response = http_client.get("/test", headers={"Authorization": f"Bearer {valid_api_key}"})
    assert response.status_code == 200
    assert response.json() == {"message": "Authentication successful"}


@patch("httpx.AsyncClient.post", new=mock_post_failure)
def test_invalid_http_authentication(http_client, invalid_api_key):
    response = http_client.get("/test", headers={"Authorization": f"Bearer {invalid_api_key}"})
    assert response.status_code == 401
    assert "Authentication failed" in response.json()["error"]["message"]


@patch("httpx.AsyncClient.post", new=mock_post_exception)
def test_http_auth_service_error(http_client, valid_api_key):
    response = http_client.get("/test", headers={"Authorization": f"Bearer {valid_api_key}"})
    assert response.status_code == 401
    assert "Authentication service error" in response.json()["error"]["message"]


def test_http_auth_request_payload(http_client, valid_api_key, mock_auth_endpoint):
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = MockResponse(200, {"message": "Authentication successful"})
        mock_post.return_value = mock_response

        http_client.get(
            "/test?param1=value1&param2=value2",
            headers={
                "Authorization": f"Bearer {valid_api_key}",
                "User-Agent": "TestClient",
                "Content-Type": "application/json",
            },
        )

        # Check that the auth endpoint was called with the correct payload
        call_args = mock_post.call_args
        assert call_args is not None

        url, kwargs = call_args[0][0], call_args[1]
        assert url == mock_auth_endpoint

        payload = kwargs["json"]
        assert payload["api_key"] == valid_api_key
        assert payload["request"]["path"] == "/test"
        assert "authorization" not in payload["request"]["headers"]
        assert "param1" in payload["request"]["params"]
        assert "param2" in payload["request"]["params"]


@pytest.mark.asyncio
async def test_http_middleware_with_access_attributes(mock_http_middleware, mock_scope):
    """Test HTTP middleware behavior with access attributes"""
    middleware, mock_app = mock_http_middleware
    mock_receive = AsyncMock()
    mock_send = AsyncMock()

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = MockResponse(
            200,
            {
                "message": "Authentication successful",
                "principal": "test-principal",
                "access_attributes": {
                    "roles": ["admin", "user"],
                    "teams": ["ml-team", "nlp-team"],
                    "projects": ["llama-3", "project-x"],
                    "namespaces": ["research", "production"],
                },
            },
        )
        mock_post.return_value = mock_response

        await middleware(mock_scope, mock_receive, mock_send)

        assert "user_attributes" in mock_scope
        attributes = mock_scope["user_attributes"]
        assert attributes["roles"] == ["admin", "user"]
        assert attributes["teams"] == ["ml-team", "nlp-team"]
        assert attributes["projects"] == ["llama-3", "project-x"]
        assert attributes["namespaces"] == ["research", "production"]

        mock_app.assert_called_once_with(mock_scope, mock_receive, mock_send)


@pytest.mark.asyncio
async def test_http_middleware_no_attributes(mock_http_middleware, mock_scope):
    """Test middleware behavior with no access attributes"""
    middleware, mock_app = mock_http_middleware
    mock_receive = AsyncMock()
    mock_send = AsyncMock()

    with patch("httpx.AsyncClient") as mock_client:
        mock_client_instance = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        mock_client_instance.post.return_value = MockResponse(
            200,
            {
                "message": "Authentication successful"
                # No access_attributes
            },
        )

        await middleware(mock_scope, mock_receive, mock_send)

        assert "user_attributes" in mock_scope
        attributes = mock_scope["user_attributes"]
        assert "roles" in attributes
        assert attributes["roles"] == ["test.jwt.token"]


# Kubernetes Tests
def test_missing_auth_header_k8s(k8s_client):
    response = k8s_client.get("/test")
    assert response.status_code == 401
    assert "Missing or invalid Authorization header" in response.json()["error"]["message"]


def test_invalid_auth_header_format_k8s(k8s_client):
    response = k8s_client.get("/test", headers={"Authorization": "InvalidFormat token123"})
    assert response.status_code == 401
    assert "Missing or invalid Authorization header" in response.json()["error"]["message"]


@patch("kubernetes.client.ApiClient")
def test_valid_k8s_authentication(mock_api_client, k8s_client, valid_token):
    # Mock the Kubernetes client
    mock_client = AsyncMock()
    mock_api_client.return_value = mock_client

    # Mock successful token validation
    mock_client.set_default_header = AsyncMock()

    # Mock the token validation to return valid access attributes
    with patch("llama_stack.distribution.server.auth_providers.KubernetesAuthProvider.validate_token") as mock_validate:
        mock_validate.return_value = TokenValidationResult(
            principal="test-principal",
            access_attributes=AccessAttributes(
                roles=["admin"], teams=["ml-team"], projects=["llama-3"], namespaces=["research"]
            ),
        )
        response = k8s_client.get("/test", headers={"Authorization": f"Bearer {valid_token}"})
        assert response.status_code == 200
        assert response.json() == {"message": "Authentication successful"}


@patch("kubernetes.client.ApiClient")
def test_invalid_k8s_authentication(mock_api_client, k8s_client, invalid_token):
    # Mock the Kubernetes client
    mock_client = AsyncMock()
    mock_api_client.return_value = mock_client

    # Mock failed token validation by raising an exception
    with patch("llama_stack.distribution.server.auth_providers.KubernetesAuthProvider.validate_token") as mock_validate:
        mock_validate.side_effect = ValueError("Invalid or expired token")
        response = k8s_client.get("/test", headers={"Authorization": f"Bearer {invalid_token}"})
        assert response.status_code == 401
        assert "Invalid or expired token" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_k8s_middleware_with_access_attributes(mock_k8s_middleware, mock_scope):
    middleware, mock_app = mock_k8s_middleware
    mock_receive = AsyncMock()
    mock_send = AsyncMock()

    with patch("kubernetes.client.ApiClient") as mock_api_client:
        mock_client = AsyncMock()
        mock_api_client.return_value = mock_client

        # Mock successful token validation
        mock_client.set_default_header = AsyncMock()

        # Mock token payload with access attributes
        mock_token_parts = ["header", "eyJzdWIiOiJhZG1pbiIsImdyb3VwcyI6WyJtbC10ZWFtIl19", "signature"]
        mock_scope["headers"][1] = (b"authorization", f"Bearer {'.'.join(mock_token_parts)}".encode())

        await middleware(mock_scope, mock_receive, mock_send)

        assert "user_attributes" in mock_scope
        assert mock_scope["user_attributes"]["roles"] == ["admin"]
        assert mock_scope["user_attributes"]["teams"] == ["ml-team"]

        mock_app.assert_called_once_with(mock_scope, mock_receive, mock_send)


@pytest.mark.asyncio
async def test_k8s_middleware_no_attributes(mock_k8s_middleware, mock_scope):
    """Test middleware behavior with no access attributes"""
    middleware, mock_app = mock_k8s_middleware
    mock_receive = AsyncMock()
    mock_send = AsyncMock()

    with patch("kubernetes.client.ApiClient") as mock_api_client:
        mock_client = AsyncMock()
        mock_api_client.return_value = mock_client

        # Mock successful token validation
        mock_client.set_default_header = AsyncMock()

        # Mock token payload without access attributes
        mock_token_parts = ["header", "eyJzdWIiOiJhZG1pbiJ9", "signature"]
        mock_scope["headers"][1] = (b"authorization", f"Bearer {'.'.join(mock_token_parts)}".encode())

        await middleware(mock_scope, mock_receive, mock_send)

        assert "user_attributes" in mock_scope
        attributes = mock_scope["user_attributes"]
        assert "roles" in attributes
        assert attributes["roles"] == ["admin"]

        mock_app.assert_called_once_with(mock_scope, mock_receive, mock_send)


# oauth2 token provider tests


@pytest.fixture
def oauth2_app():
    app = FastAPI()
    auth_config = AuthProviderConfig(
        provider_type=AuthProviderType.OAUTH2_TOKEN,
        config={
            "jwks_uri": "http://mock-authz-service/token/introspect",
            "cache_ttl": "3600",
            "audience": "llama-stack",
        },
    )
    app.add_middleware(AuthenticationMiddleware, auth_config=auth_config)

    @app.get("/test")
    def test_endpoint():
        return {"message": "Authentication successful"}

    return app


@pytest.fixture
def oauth2_client(oauth2_app):
    return TestClient(oauth2_app)


def test_missing_auth_header_oauth2(oauth2_client):
    response = oauth2_client.get("/test")
    assert response.status_code == 401
    assert "Missing or invalid Authorization header" in response.json()["error"]["message"]


def test_invalid_auth_header_format_oauth2(oauth2_client):
    response = oauth2_client.get("/test", headers={"Authorization": "InvalidFormat token123"})
    assert response.status_code == 401
    assert "Missing or invalid Authorization header" in response.json()["error"]["message"]


async def mock_jwks_response(*args, **kwargs):
    return MockResponse(
        200,
        {
            "keys": [
                {
                    "kid": "1234567890",
                    "kty": "oct",
                    "alg": "HS256",
                    "use": "sig",
                    "k": base64.b64encode(b"foobarbaz").decode(),
                }
            ]
        },
    )


@pytest.fixture
def jwt_token_valid():
    from jose import jwt

    return jwt.encode(
        {
            "sub": "my-user",
            "groups": ["group1", "group2"],
            "scope": "foo bar",
            "aud": "llama-stack",
        },
        key="foobarbaz",
        algorithm="HS256",
        headers={"kid": "1234567890"},
    )


@patch("httpx.AsyncClient.get", new=mock_jwks_response)
def test_valid_oauth2_authentication(oauth2_client, jwt_token_valid):
    response = oauth2_client.get("/test", headers={"Authorization": f"Bearer {jwt_token_valid}"})
    assert response.status_code == 200
    assert response.json() == {"message": "Authentication successful"}


@patch("httpx.AsyncClient.get", new=mock_jwks_response)
def test_invalid_oauth2_authentication(oauth2_client, invalid_token):
    response = oauth2_client.get("/test", headers={"Authorization": f"Bearer {invalid_token}"})
    assert response.status_code == 401
    assert "Invalid JWT token" in response.json()["error"]["message"]


def test_get_attributes_from_claims():
    claims = {
        "sub": "my-user",
        "groups": ["group1", "group2"],
        "scope": "foo bar",
        "aud": "llama-stack",
    }
    attributes = get_attributes_from_claims(claims, {"sub": "roles", "groups": "teams"})
    assert attributes.roles == ["my-user"]
    assert attributes.teams == ["group1", "group2"]

    claims = {
        "sub": "my-user",
        "tenant": "my-tenant",
    }
    attributes = get_attributes_from_claims(claims, {"sub": "roles", "tenant": "namespaces"})
    assert attributes.roles == ["my-user"]
    assert attributes.namespaces == ["my-tenant"]

    claims = {
        "sub": "my-user",
        "username": "my-username",
        "tenant": "my-tenant",
        "groups": ["group1", "group2"],
        "team": "my-team",
    }
    attributes = get_attributes_from_claims(
        claims,
        {
            "sub": "roles",
            "tenant": "namespaces",
            "username": "roles",
            "team": "teams",
            "groups": "teams",
        },
    )
    assert set(attributes.roles) == {"my-user", "my-username"}
    assert set(attributes.teams) == {"my-team", "group1", "group2"}
    assert attributes.namespaces == ["my-tenant"]


# TODO: add more tests for oauth2 token provider
