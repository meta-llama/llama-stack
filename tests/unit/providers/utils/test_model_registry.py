# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

#
# ModelRegistryHelper provides mixin functionality for registering and
# unregistering models. It maintains a mapping of model ID / aliases to
# provider model IDs.
#
# Test cases -
#  - Looking up an alias that does not exist should return None.
#  - Registering a model + provider ID should add the model to the registry. If provider ID is known.
#  - Registering an existing model should return an error.
#  - Unregistering a model should remove it from the registry.
#  - Unregistering a model that does not exist should return an error.
#  - Models can be registered during initialization or via register_model.
#
# Questions -
#  - Should we be allowed to register models w/o provider model IDs? No.
#    According to POST /v1/models, required params are
#      - identifier
#      - provider_resource_id
#      - provider_id
#      - type
#      - metadata
#      - model_type
#
#  TODO: llama_model functionality
#

import pytest

from llama_stack.apis.models.models import Model
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper, ProviderModelEntry


@pytest.fixture
def known_model() -> Model:
    return Model(
        provider_id="provider",
        identifier="known-model",
        provider_resource_id="known-provider-id",
    )


@pytest.fixture
def known_provider_model(known_model: Model) -> ProviderModelEntry:
    return ProviderModelEntry(
        provider_model_id=known_model.provider_resource_id,
        # aliases=[],
    )


@pytest.fixture
def unknown_model() -> Model:
    return Model(
        provider_id="provider",
        identifier="unknown-model",
        provider_resource_id="unknown-provider-id",
    )


@pytest.fixture
def helper(known_provider_model: ProviderModelEntry) -> ModelRegistryHelper:
    return ModelRegistryHelper([known_provider_model])


@pytest.mark.asyncio
async def test_lookup_unknown_model(helper: ModelRegistryHelper, unknown_model: Model) -> None:
    assert helper.get_provider_model_id(unknown_model.model_id) is None


@pytest.mark.asyncio
async def test_register_unknown_provider_model(helper: ModelRegistryHelper, unknown_model: Model) -> None:
    with pytest.raises(ValueError):
        await helper.register_model(unknown_model)


@pytest.mark.asyncio
async def test_register_model(helper: ModelRegistryHelper, known_model: Model) -> None:
    assert helper.get_provider_model_id(known_model.model_id) is None
    await helper.register_model(known_model)
    assert helper.get_provider_model_id(known_model.model_id) == known_model.provider_resource_id


@pytest.mark.asyncio
async def test_register_model_existing(helper: ModelRegistryHelper, known_model: Model) -> None:
    await helper.register_model(known_model)
    with pytest.raises(ValueError):
        await helper.register_model(known_model)


@pytest.mark.asyncio
async def test_unregister_model(helper: ModelRegistryHelper, known_model: Model) -> None:
    await helper.register_model(known_model)
    assert helper.get_provider_model_id(known_model.model_id) == known_model.provider_model_id
    await helper.unregister_model(known_model.model_id)
    assert helper.get_provider_model_id(known_model.model_id) is None


@pytest.mark.asyncio
async def test_unregister_unknown_model(helper: ModelRegistryHelper, unknown_model: Model) -> None:
    with pytest.raises(ValueError):
        await helper.unregister_model(unknown_model.model_id)


@pytest.mark.asyncio
async def test_register_model_during_init(helper: ModelRegistryHelper, known_model: Model) -> None:
    assert helper.get_provider_model_id(known_model.provider_resource_id) == known_model.provider_model_id


@pytest.mark.asyncio
async def test_register_model_existing_from_init(helper: ModelRegistryHelper, known_model: Model) -> None:
    assert helper.get_provider_model_id(known_model.provider_resource_id) == known_model.provider_model_id
    with pytest.raises(ValueError):
        known_model.identifier = known_model.provider_resource_id
        await helper.register_model(known_model)


@pytest.mark.asyncio
async def test_unregister_model_during_init(helper: ModelRegistryHelper, known_model: Model) -> None:
    assert helper.get_provider_model_id(known_model.provider_resource_id) == known_model.provider_model_id
    await helper.unregister_model(known_model.provider_resource_id)
    assert helper.get_provider_model_id(known_model.provider_resource_id) is None
