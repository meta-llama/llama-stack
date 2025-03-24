# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import shutil
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

from llama_stack.apis.datatypes import Api
from llama_stack.apis.models import ModelType
from llama_stack.distribution.datatypes import AccessAttributes, ModelWithACL
from llama_stack.distribution.routers.routing_tables import ModelsRoutingTable
from llama_stack.distribution.store.registry import CachedDiskDistributionRegistry
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig
from llama_stack.providers.utils.kvstore.sqlite import SqliteKVStoreImpl


class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


def _return_model(model):
    return model


@pytest.fixture
async def test_setup():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_access_control.db")
    kvstore_config = SqliteKVStoreConfig(db_path=db_path)
    kvstore = SqliteKVStoreImpl(kvstore_config)
    await kvstore.initialize()
    registry = CachedDiskDistributionRegistry(kvstore)
    await registry.initialize()

    mock_inference = Mock()
    mock_inference.__provider_spec__ = MagicMock()
    mock_inference.__provider_spec__.api = Api.inference
    mock_inference.register_model = AsyncMock(side_effect=_return_model)
    routing_table = ModelsRoutingTable(
        impls_by_provider_id={"test_provider": mock_inference},
        dist_registry=registry,
    )
    yield registry, routing_table
    shutil.rmtree(temp_dir)


@pytest.mark.asyncio
@patch("llama_stack.distribution.routers.routing_tables.get_auth_attributes")
async def test_access_control_with_cache(mock_get_auth_attributes, test_setup):
    registry, routing_table = test_setup
    model_public = ModelWithACL(
        identifier="model-public",
        provider_id="test_provider",
        provider_resource_id="model-public",
        model_type=ModelType.llm,
    )
    model_admin_only = ModelWithACL(
        identifier="model-admin",
        provider_id="test_provider",
        provider_resource_id="model-admin",
        model_type=ModelType.llm,
        access_attributes=AccessAttributes(roles=["admin"]),
    )
    model_data_scientist = ModelWithACL(
        identifier="model-data-scientist",
        provider_id="test_provider",
        provider_resource_id="model-data-scientist",
        model_type=ModelType.llm,
        access_attributes=AccessAttributes(roles=["data-scientist", "researcher"], teams=["ml-team"]),
    )
    await registry.register(model_public)
    await registry.register(model_admin_only)
    await registry.register(model_data_scientist)

    mock_get_auth_attributes.return_value = {"roles": ["admin"], "teams": ["management"]}
    all_models = await routing_table.list_models()
    assert len(all_models.data) == 2

    model = await routing_table.get_model("model-public")
    assert model.identifier == "model-public"
    model = await routing_table.get_model("model-admin")
    assert model.identifier == "model-admin"
    with pytest.raises(ValueError):
        await routing_table.get_model("model-data-scientist")

    mock_get_auth_attributes.return_value = {"roles": ["data-scientist"], "teams": ["other-team"]}
    all_models = await routing_table.list_models()
    assert len(all_models.data) == 1
    assert all_models.data[0].identifier == "model-public"
    model = await routing_table.get_model("model-public")
    assert model.identifier == "model-public"
    with pytest.raises(ValueError):
        await routing_table.get_model("model-admin")
    with pytest.raises(ValueError):
        await routing_table.get_model("model-data-scientist")

    mock_get_auth_attributes.return_value = {"roles": ["data-scientist"], "teams": ["ml-team"]}
    all_models = await routing_table.list_models()
    assert len(all_models.data) == 2
    model_ids = [m.identifier for m in all_models.data]
    assert "model-public" in model_ids
    assert "model-data-scientist" in model_ids
    assert "model-admin" not in model_ids
    model = await routing_table.get_model("model-public")
    assert model.identifier == "model-public"
    model = await routing_table.get_model("model-data-scientist")
    assert model.identifier == "model-data-scientist"
    with pytest.raises(ValueError):
        await routing_table.get_model("model-admin")


@pytest.mark.asyncio
@patch("llama_stack.distribution.routers.routing_tables.get_auth_attributes")
async def test_access_control_and_updates(mock_get_auth_attributes, test_setup):
    registry, routing_table = test_setup
    model_public = ModelWithACL(
        identifier="model-updates",
        provider_id="test_provider",
        provider_resource_id="model-updates",
        model_type=ModelType.llm,
    )
    await registry.register(model_public)
    mock_get_auth_attributes.return_value = {
        "roles": ["user"],
    }
    model = await routing_table.get_model("model-updates")
    assert model.identifier == "model-updates"
    model_public.access_attributes = AccessAttributes(roles=["admin"])
    await registry.update(model_public)
    mock_get_auth_attributes.return_value = {
        "roles": ["user"],
    }
    with pytest.raises(ValueError):
        await routing_table.get_model("model-updates")
    mock_get_auth_attributes.return_value = {
        "roles": ["admin"],
    }
    model = await routing_table.get_model("model-updates")
    assert model.identifier == "model-updates"


@pytest.mark.asyncio
@patch("llama_stack.distribution.routers.routing_tables.get_auth_attributes")
async def test_access_control_empty_attributes(mock_get_auth_attributes, test_setup):
    registry, routing_table = test_setup
    model = ModelWithACL(
        identifier="model-empty-attrs",
        provider_id="test_provider",
        provider_resource_id="model-empty-attrs",
        model_type=ModelType.llm,
        access_attributes=AccessAttributes(),
    )
    await registry.register(model)
    mock_get_auth_attributes.return_value = {
        "roles": [],
    }
    result = await routing_table.get_model("model-empty-attrs")
    assert result.identifier == "model-empty-attrs"
    all_models = await routing_table.list_models()
    model_ids = [m.identifier for m in all_models.data]
    assert "model-empty-attrs" in model_ids


@pytest.mark.asyncio
@patch("llama_stack.distribution.routers.routing_tables.get_auth_attributes")
async def test_no_user_attributes(mock_get_auth_attributes, test_setup):
    registry, routing_table = test_setup
    model_public = ModelWithACL(
        identifier="model-public-2",
        provider_id="test_provider",
        provider_resource_id="model-public-2",
        model_type=ModelType.llm,
    )
    model_restricted = ModelWithACL(
        identifier="model-restricted",
        provider_id="test_provider",
        provider_resource_id="model-restricted",
        model_type=ModelType.llm,
        access_attributes=AccessAttributes(roles=["admin"]),
    )
    await registry.register(model_public)
    await registry.register(model_restricted)
    mock_get_auth_attributes.return_value = None
    model = await routing_table.get_model("model-public-2")
    assert model.identifier == "model-public-2"

    with pytest.raises(ValueError):
        await routing_table.get_model("model-restricted")

    all_models = await routing_table.list_models()
    assert len(all_models.data) == 1
    assert all_models.data[0].identifier == "model-public-2"


@pytest.mark.asyncio
@patch("llama_stack.distribution.routers.routing_tables.get_auth_attributes")
async def test_automatic_access_attributes(mock_get_auth_attributes, test_setup):
    """Test that newly created resources inherit access attributes from their creator."""
    registry, routing_table = test_setup

    # Set creator's attributes
    creator_attributes = {"roles": ["data-scientist"], "teams": ["ml-team"], "projects": ["llama-3"]}
    mock_get_auth_attributes.return_value = creator_attributes

    # Create model without explicit access attributes
    model = ModelWithACL(
        identifier="auto-access-model",
        provider_id="test_provider",
        provider_resource_id="auto-access-model",
        model_type=ModelType.llm,
    )
    await routing_table.register_object(model)

    # Verify the model got creator's attributes
    registered_model = await routing_table.get_model("auto-access-model")
    assert registered_model.access_attributes is not None
    assert registered_model.access_attributes.roles == ["data-scientist"]
    assert registered_model.access_attributes.teams == ["ml-team"]
    assert registered_model.access_attributes.projects == ["llama-3"]

    # Verify another user without matching attributes can't access it
    mock_get_auth_attributes.return_value = {"roles": ["engineer"], "teams": ["infra-team"]}
    with pytest.raises(ValueError):
        await routing_table.get_model("auto-access-model")

    # But a user with matching attributes can
    mock_get_auth_attributes.return_value = {
        "roles": ["data-scientist", "engineer"],
        "teams": ["ml-team", "platform-team"],
        "projects": ["llama-3"],
    }
    model = await routing_table.get_model("auto-access-model")
    assert model.identifier == "auto-access-model"
