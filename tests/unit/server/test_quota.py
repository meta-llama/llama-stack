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

from llama_stack.distribution.server.quota import QuotaMiddleware
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig


@pytest.fixture(autouse=True)
def clean_sqlite_db():
    """
    Remove the quotas.db file before each test to ensure no leftover state on disk.
    """
    db_path = "./quotas_test.db"
    if os.path.exists(db_path):
        os.remove(db_path)


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


@pytest.fixture(scope="function")
def app(request):
    """
    Create a FastAPI app with both InjectClientIDMiddleware and QuotaMiddleware.
    Each test gets a unique client_id for safety.
    """
    inner_app = FastAPI()

    @inner_app.get("/test")
    async def test_endpoint():
        return {"message": "ok"}

    # Use the test name to create a unique client_id per test
    client_id = f"client_{request.node.name}"

    app = InjectClientIDMiddleware(
        QuotaMiddleware(
            inner_app,
            kv_config=SqliteKVStoreConfig(db_path="./quotas_test.db"),
            max_requests=2,
            window_seconds=60,
        ),
        client_id=client_id,
    )

    return app


def test_quota_allows_up_to_limit(app):
    client = TestClient(app)

    resp1 = client.get("/test")
    assert resp1.status_code == 200
    assert resp1.json() == {"message": "ok"}

    resp2 = client.get("/test")
    assert resp2.status_code == 200
    assert resp2.json() == {"message": "ok"}


def test_quota_blocks_after_limit(app):
    client = TestClient(app)

    # Exceed limit: 3rd request should be throttled
    client.get("/test")
    client.get("/test")
    resp3 = client.get("/test")
    assert resp3.status_code == 429
    assert resp3.json()["error"]["message"] == "Quota exceeded"


def test_missing_authenticated_client_id_returns_500():
    """
    Confirm 500 error when QuotaMiddleware runs without authenticated_client_id.
    """
    inner_app = FastAPI()

    @inner_app.get("/test")
    async def test_endpoint():
        return {"message": "ok"}

    test_app = QuotaMiddleware(
        inner_app,
        kv_config=SqliteKVStoreConfig(db_path="./quotas_test.db"),
        max_requests=2,
        window_seconds=60,
    )

    client = TestClient(test_app)

    resp = client.get("/test")
    assert resp.status_code == 500
    assert "Quota system misconfigured" in resp.json()["error"]["message"]
