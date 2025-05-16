# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from llama_stack.distribution.datatypes import QuotaConfig, QuotaPeriod
from llama_stack.distribution.server.quota import QuotaMiddleware
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig

TEST_DB_PATH = "./quotas_test.db"


@pytest.fixture(autouse=True)
def clean_sqlite_db():
    """
    Remove the test DB file before each test to ensure clean state.
    """
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


class InjectClientIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that injects 'authenticated_client_id' to mimic AuthenticationMiddleware.
    """

    def __init__(self, app, client_id="client1"):
        super().__init__(app)
        self.client_id = client_id

    async def dispatch(self, request: Request, call_next):
        request.scope["authenticated_client_id"] = self.client_id
        return await call_next(request)


def build_quota_config() -> QuotaConfig:
    return QuotaConfig(
        kvstore=SqliteKVStoreConfig(db_path=TEST_DB_PATH),
        anonymous_max_requests=1,
        authenticated_max_requests=2,
        period=QuotaPeriod.DAY,
    )


@pytest.fixture(scope="function")
def auth_app(request):
    """
    FastAPI app with InjectClientIDMiddleware and QuotaMiddleware for authenticated testing.
    """
    inner_app = FastAPI()

    @inner_app.get("/test")
    async def test_endpoint():
        return {"message": "ok"}

    client_id = f"client_{request.node.name}"
    quota = build_quota_config()

    app = InjectClientIDMiddleware(
        QuotaMiddleware(
            inner_app,
            kv_config=quota.kvstore,
            anonymous_max_requests=quota.anonymous_max_requests,
            authenticated_max_requests=quota.authenticated_max_requests,
            window_seconds=86400,
        ),
        client_id=client_id,
    )
    return app


def test_authenticated_quota_allows_up_to_limit(auth_app):
    client = TestClient(auth_app)
    assert client.get("/test").status_code == 200
    assert client.get("/test").status_code == 200


def test_authenticated_quota_blocks_after_limit(auth_app):
    client = TestClient(auth_app)
    client.get("/test")
    client.get("/test")
    resp = client.get("/test")
    assert resp.status_code == 429
    assert resp.json()["error"]["message"] == "Quota exceeded"


def test_anonymous_quota_allows_up_to_limit():
    inner_app = FastAPI()

    @inner_app.get("/test")
    async def test_endpoint():
        return {"message": "ok"}

    quota = build_quota_config()

    app = QuotaMiddleware(
        inner_app,
        kv_config=quota.kvstore,
        anonymous_max_requests=quota.anonymous_max_requests,
        authenticated_max_requests=quota.authenticated_max_requests,
        window_seconds=86400,
    )

    client = TestClient(app)
    assert client.get("/test").status_code == 200


def test_anonymous_quota_blocks_after_limit():
    inner_app = FastAPI()

    @inner_app.get("/test")
    async def test_endpoint():
        return {"message": "ok"}

    quota = build_quota_config()

    app = QuotaMiddleware(
        inner_app,
        kv_config=quota.kvstore,
        anonymous_max_requests=quota.anonymous_max_requests,
        authenticated_max_requests=quota.authenticated_max_requests,
        window_seconds=86400,
    )

    client = TestClient(app)
    client.get("/test")
    resp = client.get("/test")
    assert resp.status_code == 429
    assert resp.json()["error"]["message"] == "Quota exceeded"
