# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from llama_stack.core.datatypes import CORSConfig, process_cors_config


class TestCORSConfig:
    """Test basic CORS configuration."""

    def test_defaults(self):
        config = CORSConfig()

        assert config.allow_origins == ["*"]
        assert config.allow_origin_regex is None
        assert config.allow_methods == ["*"]
        assert config.allow_headers == ["*"]
        assert config.allow_credentials is False
        assert config.expose_headers == []
        assert config.max_age == 600

    def test_custom_values(self):
        config = CORSConfig(allow_origins=["https://example.com"], allow_credentials=True, max_age=3600)

        assert config.allow_origins == ["https://example.com"]
        assert config.allow_credentials is True
        assert config.max_age == 3600

    def test_regex_field(self):
        config = CORSConfig(allow_origins=[], allow_origin_regex=r"https?://localhost:\d+")

        assert config.allow_origins == []
        assert config.allow_origin_regex == r"https?://localhost:\d+"

    def test_credentials_with_wildcard_error(self):
        """Should raise error when using credentials with wildcard origins."""
        with pytest.raises(ValueError, match="CORS: allow_credentials=True requires explicit origins"):
            CORSConfig(allow_origins=["*"], allow_credentials=True)


class TestProcessCORSConfig:
    """Test the process_cors_config function."""

    def test_none_returns_none(self):
        result = process_cors_config(None)
        assert result is None

    def test_false_returns_none(self):
        result = process_cors_config(False)
        assert result is None

    def test_true_returns_dev_config(self):
        """Test dev mode: cors: true"""
        result = process_cors_config(True)

        assert isinstance(result, CORSConfig)
        assert result.allow_origins == []
        assert result.allow_origin_regex == r"https?://localhost:\d+"
        assert result.allow_credentials is False
        assert "GET" in result.allow_methods
        assert "POST" in result.allow_methods

    def test_cors_object_returned_as_is(self):
        original = CORSConfig(allow_origins=["https://example.com"])
        result = process_cors_config(original)

        assert result is original

    def test_invalid_type_raises_error(self):
        with pytest.raises(ValueError, match="Invalid CORS configuration type"):
            process_cors_config("invalid")


class TestCORSIntegration:
    """Test CORS with FastAPI integration."""

    def test_dev_mode_with_fastapi(self):
        """Test that dev mode config works with FastAPI middleware."""
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.testclient import TestClient

        app = FastAPI()

        # Use our dev mode config
        cors_config = process_cors_config(True)
        app.add_middleware(CORSMiddleware, **cors_config.model_dump())

        @app.get("/test")
        def test_endpoint():
            return {"message": "hello"}

        client = TestClient(app)

        # Test localhost origins work
        response = client.get("/test", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 200
        assert response.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"

        # Test non-localhost doesn't get CORS headers
        response = client.get("/test", headers={"Origin": "https://evil.com"})
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" not in response.headers

    def test_production_mode_with_fastapi(self):
        """Test explicit origins configuration."""
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.testclient import TestClient

        app = FastAPI()

        # Production config
        cors_config = CORSConfig(allow_origins=["https://myapp.com"], allow_credentials=True)
        app.add_middleware(CORSMiddleware, **cors_config.model_dump())

        @app.get("/test")
        def test_endpoint():
            return {"message": "hello"}

        client = TestClient(app)

        # Test allowed origin works
        response = client.get("/test", headers={"Origin": "https://myapp.com"})
        assert response.status_code == 200
        assert response.headers.get("Access-Control-Allow-Origin") == "https://myapp.com"
        assert response.headers.get("Access-Control-Allow-Credentials") == "true"

        # Test disallowed origin
        response = client.get("/test", headers={"Origin": "https://evil.com"})
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" not in response.headers

    def test_preflight_request(self):
        """Test CORS preflight OPTIONS request."""
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.testclient import TestClient

        app = FastAPI()

        cors_config = process_cors_config(True)
        app.add_middleware(CORSMiddleware, **cors_config.model_dump())

        @app.get("/test")
        def test_endpoint():
            return {"message": "hello"}

        client = TestClient(app)

        # Preflight request
        response = client.options(
            "/test", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"}
        )

        assert response.status_code == 200
        assert response.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
        assert "GET" in response.headers.get("Access-Control-Allow-Methods", "")
