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


async def mock_post_success(*args, **kwargs):
    mock_response = AsyncMock()
    mock_response.status_code = 200
    return mock_response


async def mock_post_failure(*args, **kwargs):
    mock_response = AsyncMock()
    mock_response.status_code = 401
    return mock_response


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
        mock_response = AsyncMock()
        mock_response.status_code = 200
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
        assert "authorization" in payload["request"]["headers"]
        assert "param1" in payload["request"]["params"]
        assert "param2" in payload["request"]["params"]
