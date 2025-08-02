# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest

from llama_stack.apis.common.errors import MissingEmbeddingModelError
from llama_stack.apis.models import ModelType
from llama_stack.core.routers.vector_io import VectorIORouter


class _DummyModel:
    def __init__(self, identifier: str, dim: int):
        self.identifier = identifier
        self.model_type = ModelType.embedding
        self.metadata = {"embedding_dimension": dim}


class _DummyRoutingTable:
    """Minimal stub satisfying the methods used by VectorIORouter in tests."""

    def __init__(self):
        self._models: list[_DummyModel] = [
            _DummyModel("first-model", 123),
            _DummyModel("second-model", 512),
        ]

    async def get_all_with_type(self, _type: str):
        # Only embedding models requested in our tests
        return self._models

    # The following methods are required by the VectorIORouter signature but
    # are not used in these unit tests; stub them out.
    async def register_vector_db(self, *args, **kwargs):
        raise NotImplementedError

    async def get_provider_impl(self, *args, **kwargs):
        raise NotImplementedError


async def test_global_default_used(monkeypatch):
    """Router should pick up global default when no explicit model is supplied."""

    monkeypatch.setenv("LLAMA_STACK_DEFAULT_EMBEDDING_MODEL", "env-default-model")
    monkeypatch.setenv("LLAMA_STACK_DEFAULT_EMBEDDING_DIMENSION", "256")

    router = VectorIORouter(routing_table=_DummyRoutingTable())

    model, dim = await router._resolve_embedding_model(None)
    assert model == "env-default-model"
    assert dim == 256

    # Cleanup env vars
    monkeypatch.delenv("LLAMA_STACK_DEFAULT_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("LLAMA_STACK_DEFAULT_EMBEDDING_DIMENSION", raising=False)


async def test_explicit_override(monkeypatch):
    """Explicit model parameter should override global default."""

    monkeypatch.setenv("LLAMA_STACK_DEFAULT_EMBEDDING_MODEL", "env-default-model")

    router = VectorIORouter(routing_table=_DummyRoutingTable())

    model, dim = await router._resolve_embedding_model("first-model")
    assert model == "first-model"
    assert dim == 123

    monkeypatch.delenv("LLAMA_STACK_DEFAULT_EMBEDDING_MODEL", raising=False)


async def test_error_when_no_default():
    """Router should raise when neither explicit nor global default is available."""

    router = VectorIORouter(routing_table=_DummyRoutingTable())

    with pytest.raises(MissingEmbeddingModelError):
        await router._resolve_embedding_model(None)
