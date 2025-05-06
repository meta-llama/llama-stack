# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest

from llama_stack.apis.models import ModelType
from llama_stack.distribution.datatypes import ModelWithACL
from llama_stack.distribution.server.auth_providers import AccessAttributes
from llama_stack.distribution.store.registry import CachedDiskDistributionRegistry


@pytest.mark.asyncio
async def test_registry_cache_with_acl(cached_disk_dist_registry):
    model = ModelWithACL(
        identifier="model-acl",
        provider_id="test-provider",
        provider_resource_id="model-acl-resource",
        model_type=ModelType.llm,
        access_attributes=AccessAttributes(roles=["admin"], teams=["ai-team"]),
    )

    success = await cached_disk_dist_registry.register(model)
    assert success

    cached_model = cached_disk_dist_registry.get_cached("model", "model-acl")
    assert cached_model is not None
    assert cached_model.identifier == "model-acl"
    assert cached_model.access_attributes.roles == ["admin"]
    assert cached_model.access_attributes.teams == ["ai-team"]

    fetched_model = await cached_disk_dist_registry.get("model", "model-acl")
    assert fetched_model is not None
    assert fetched_model.identifier == "model-acl"
    assert fetched_model.access_attributes.roles == ["admin"]

    model.access_attributes = AccessAttributes(roles=["admin", "user"], projects=["project-x"])
    await cached_disk_dist_registry.update(model)

    updated_cached = cached_disk_dist_registry.get_cached("model", "model-acl")
    assert updated_cached is not None
    assert updated_cached.access_attributes.roles == ["admin", "user"]
    assert updated_cached.access_attributes.projects == ["project-x"]
    assert updated_cached.access_attributes.teams is None

    new_registry = CachedDiskDistributionRegistry(cached_disk_dist_registry.kvstore)
    await new_registry.initialize()

    new_model = await new_registry.get("model", "model-acl")
    assert new_model is not None
    assert new_model.identifier == "model-acl"
    assert new_model.access_attributes.roles == ["admin", "user"]
    assert new_model.access_attributes.projects == ["project-x"]
    assert new_model.access_attributes.teams is None


@pytest.mark.asyncio
async def test_registry_empty_acl(cached_disk_dist_registry):
    model = ModelWithACL(
        identifier="model-empty-acl",
        provider_id="test-provider",
        provider_resource_id="model-resource",
        model_type=ModelType.llm,
        access_attributes=AccessAttributes(),
    )

    await cached_disk_dist_registry.register(model)

    cached_model = cached_disk_dist_registry.get_cached("model", "model-empty-acl")
    assert cached_model is not None
    assert cached_model.access_attributes is not None
    assert cached_model.access_attributes.roles is None
    assert cached_model.access_attributes.teams is None
    assert cached_model.access_attributes.projects is None
    assert cached_model.access_attributes.namespaces is None

    all_models = await cached_disk_dist_registry.get_all()
    assert len(all_models) == 1

    model = ModelWithACL(
        identifier="model-no-acl",
        provider_id="test-provider",
        provider_resource_id="model-resource-2",
        model_type=ModelType.llm,
    )

    await cached_disk_dist_registry.register(model)

    cached_model = cached_disk_dist_registry.get_cached("model", "model-no-acl")
    assert cached_model is not None
    assert cached_model.access_attributes is None

    all_models = await cached_disk_dist_registry.get_all()
    assert len(all_models) == 2


@pytest.mark.asyncio
async def test_registry_serialization(cached_disk_dist_registry):
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

    await cached_disk_dist_registry.register(model)

    new_registry = CachedDiskDistributionRegistry(cached_disk_dist_registry.kvstore)
    await new_registry.initialize()

    loaded_model = await new_registry.get("model", "model-serialize")
    assert loaded_model is not None

    assert loaded_model.access_attributes.roles == ["admin", "researcher"]
    assert loaded_model.access_attributes.teams == ["ai-team", "ml-team"]
    assert loaded_model.access_attributes.projects == ["project-a", "project-b"]
    assert loaded_model.access_attributes.namespaces == ["prod", "staging"]
