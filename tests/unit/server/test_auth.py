# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llama_stack.distribution.server.auth import AuthenticationMiddleware


class MockResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data


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
def app(mock_auth_endpoint):
    app = FastAPI()
    app.add_middleware(AuthenticationMiddleware, auth_endpoint=mock_auth_endpoint)

    @app.get("/test")
    def test_endpoint():
        return {"message": "Authentication successful"}

    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def mock_scope():
    return {
        "type": "http",
        "path": "/models/list",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer test-api-key"),
            (b"user-agent", b"test-user-agent"),
        ],
        "query_string": b"limit=100&offset=0",
    }


@pytest.fixture
def mock_middleware(mock_auth_endpoint):
    mock_app = AsyncMock()
    return AuthenticationMiddleware(mock_app, mock_auth_endpoint), mock_app


async def mock_post_success(*args, **kwargs):
    return MockResponse(200, {"message": "Authentication successful"})


async def mock_post_failure(*args, **kwargs):
    return MockResponse(401, {"message": "Authentication failed"})


async def mock_post_exception(*args, **kwargs):
    raise Exception("Connection error")


def test_missing_auth_header(client):
    response = client.get("/test")
    assert response.status_code == 401
    assert "Missing or invalid Authorization header" in response.json()["error"]["message"]


def test_invalid_auth_header_format(client):
    response = client.get("/test", headers={"Authorization": "InvalidFormat token123"})
    assert response.status_code == 401
    assert "Missing or invalid Authorization header" in response.json()["error"]["message"]


@patch("httpx.AsyncClient.post", new=mock_post_success)
def test_valid_authentication(client, valid_api_key):
    response = client.get("/test", headers={"Authorization": f"Bearer {valid_api_key}"})
    assert response.status_code == 200
    assert response.json() == {"message": "Authentication successful"}


@patch("httpx.AsyncClient.post", new=mock_post_failure)
def test_invalid_authentication(client, invalid_api_key):
    response = client.get("/test", headers={"Authorization": f"Bearer {invalid_api_key}"})
    assert response.status_code == 401
    assert "Authentication failed" in response.json()["error"]["message"]


@patch("httpx.AsyncClient.post", new=mock_post_exception)
def test_auth_service_error(client, valid_api_key):
    response = client.get("/test", headers={"Authorization": f"Bearer {valid_api_key}"})
    assert response.status_code == 401
    assert "Authentication service error" in response.json()["error"]["message"]


def test_auth_request_payload(client, valid_api_key, mock_auth_endpoint):
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = MockResponse(200, {"message": "Authentication successful"})
        mock_post.return_value = mock_response

        client.get(
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
async def test_auth_middleware_with_access_attributes(mock_middleware, mock_scope):
    middleware, mock_app = mock_middleware
    mock_receive = AsyncMock()
    mock_send = AsyncMock()

    with patch("httpx.AsyncClient") as mock_client:
        mock_client_instance = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        mock_client_instance.post.return_value = MockResponse(
            200,
            {
                "access_attributes": {
                    "roles": ["admin", "user"],
                    "teams": ["ml-team"],
                    "projects": ["project-x", "project-y"],
                }
            },
        )

        await middleware(mock_scope, mock_receive, mock_send)

        assert "user_attributes" in mock_scope
        assert mock_scope["user_attributes"]["roles"] == ["admin", "user"]
        assert mock_scope["user_attributes"]["teams"] == ["ml-team"]
        assert mock_scope["user_attributes"]["projects"] == ["project-x", "project-y"]

        mock_app.assert_called_once_with(mock_scope, mock_receive, mock_send)


@pytest.mark.asyncio
async def test_auth_middleware_no_attributes(mock_middleware, mock_scope):
    """Test middleware behavior with no access attributes"""
    middleware, mock_app = mock_middleware
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
        assert "namespaces" in attributes
        assert attributes["namespaces"] == ["test-api-key"]
