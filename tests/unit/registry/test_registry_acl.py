# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import shutil
import tempfile

import pytest

from llama_stack.apis.models import ModelType
from llama_stack.distribution.datatypes import ModelWithACL
from llama_stack.distribution.server.auth import AccessAttributes
from llama_stack.distribution.store.registry import CachedDiskDistributionRegistry
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig
from llama_stack.providers.utils.kvstore.sqlite import SqliteKVStoreImpl


@pytest.fixture(scope="function")
async def kvstore():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_registry_acl.db")
    kvstore_config = SqliteKVStoreConfig(db_path=db_path)
    kvstore = SqliteKVStoreImpl(kvstore_config)
    await kvstore.initialize()
    yield kvstore
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
async def registry(kvstore):
    registry = CachedDiskDistributionRegistry(kvstore)
    await registry.initialize()
    return registry


@pytest.mark.asyncio
async def test_registry_cache_with_acl(registry):
    model = ModelWithACL(
        identifier="model-acl",
        provider_id="test-provider",
        provider_resource_id="model-acl-resource",
        model_type=ModelType.llm,
        access_attributes=AccessAttributes(roles=["admin"], teams=["ai-team"]),
    )

    success = await registry.register(model)
    assert success

    cached_model = registry.get_cached("model", "model-acl")
    assert cached_model is not None
    assert cached_model.identifier == "model-acl"
    assert cached_model.access_attributes.roles == ["admin"]
    assert cached_model.access_attributes.teams == ["ai-team"]

    fetched_model = await registry.get("model", "model-acl")
    assert fetched_model is not None
    assert fetched_model.identifier == "model-acl"
    assert fetched_model.access_attributes.roles == ["admin"]

    model.access_attributes = AccessAttributes(roles=["admin", "user"], projects=["project-x"])
    await registry.update(model)

    updated_cached = registry.get_cached("model", "model-acl")
    assert updated_cached is not None
    assert updated_cached.access_attributes.roles == ["admin", "user"]
    assert updated_cached.access_attributes.projects == ["project-x"]
    assert updated_cached.access_attributes.teams is None

    new_registry = CachedDiskDistributionRegistry(registry.kvstore)
    await new_registry.initialize()

    new_model = await new_registry.get("model", "model-acl")
    assert new_model is not None
    assert new_model.identifier == "model-acl"
    assert new_model.access_attributes.roles == ["admin", "user"]
    assert new_model.access_attributes.projects == ["project-x"]
    assert new_model.access_attributes.teams is None


@pytest.mark.asyncio
async def test_registry_empty_acl(registry):
    model = ModelWithACL(
        identifier="model-empty-acl",
        provider_id="test-provider",
        provider_resource_id="model-resource",
        model_type=ModelType.llm,
        access_attributes=AccessAttributes(),
    )

    await registry.register(model)

    cached_model = registry.get_cached("model", "model-empty-acl")
    assert cached_model is not None
    assert cached_model.access_attributes is not None
    assert cached_model.access_attributes.roles is None
    assert cached_model.access_attributes.teams is None
    assert cached_model.access_attributes.projects is None
    assert cached_model.access_attributes.namespaces is None

    all_models = await registry.get_all()
    assert len(all_models) == 1

    model = ModelWithACL(
        identifier="model-no-acl",
        provider_id="test-provider",
        provider_resource_id="model-resource-2",
        model_type=ModelType.llm,
    )

    await registry.register(model)

    cached_model = registry.get_cached("model", "model-no-acl")
    assert cached_model is not None
    assert cached_model.access_attributes is None

    all_models = await registry.get_all()
    assert len(all_models) == 2


@pytest.mark.asyncio
async def test_registry_serialization(registry):
    attributes = AccessAttributes(
        roles=["admin", "researcher"],
        teams=["ai-team", "ml-team"],
        projects=["project-a", "project-b"],
        namespaces=["prod", "staging"],
    )

    model = ModelWithACL(
        identifier="model-serialize",
        provider_id="test-provider",
        provider_resource_id="model-resource",
        model_type=ModelType.llm,
        access_attributes=attributes,
    )

    await registry.register(model)

    new_registry = CachedDiskDistributionRegistry(registry.kvstore)
    await new_registry.initialize()

    loaded_model = await new_registry.get("model", "model-serialize")
    assert loaded_model is not None

    assert loaded_model.access_attributes.roles == ["admin", "researcher"]
    assert loaded_model.access_attributes.teams == ["ai-team", "ml-team"]
    assert loaded_model.access_attributes.projects == ["project-a", "project-b"]
    assert loaded_model.access_attributes.namespaces == ["prod", "staging"]
