# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import tempfile

import pytest
import pytest_asyncio

from llama_stack.apis.models import ModelInput
from llama_stack.distribution.datatypes import Api, Provider

from llama_stack.providers.inline.agents.meta_reference import (
    MetaReferenceAgentsImplConfig,
)

from llama_stack.providers.tests.resolver import construct_stack_for_test
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig
from ..conftest import ProviderFixture, remote_stack_fixture


def pick_inference_model(inference_model):
    # This is not entirely satisfactory. The fixture `inference_model` can correspond to
    # multiple models when you need to run a safety model in addition to normal agent
    # inference model. We filter off the safety model by looking for "Llama-Guard"
    if isinstance(inference_model, list):
        inference_model = next(m for m in inference_model if "Llama-Guard" not in m)
        assert inference_model is not None
    return inference_model


@pytest.fixture(scope="session")
def agents_remote() -> ProviderFixture:
    return remote_stack_fixture()


@pytest.fixture(scope="session")
def agents_meta_reference() -> ProviderFixture:
    sqlite_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="meta-reference",
                provider_type="inline::meta-reference",
                config=MetaReferenceAgentsImplConfig(
                    # TODO: make this an in-memory store
                    persistence_store=SqliteKVStoreConfig(
                        db_path=sqlite_file.name,
                    ),
                ).model_dump(),
            )
        ],
    )


AGENTS_FIXTURES = ["meta_reference", "remote"]


@pytest_asyncio.fixture(scope="session")
async def agents_stack(request, inference_model, safety_shield):
    fixture_dict = request.param

    providers = {}
    provider_data = {}
    for key in ["inference", "safety", "memory", "agents"]:
        fixture = request.getfixturevalue(f"{key}_{fixture_dict[key]}")
        providers[key] = fixture.providers
        if fixture.provider_data:
            provider_data.update(fixture.provider_data)

    inference_models = (
        inference_model if isinstance(inference_model, list) else [inference_model]
    )
    test_stack = await construct_stack_for_test(
        [Api.agents, Api.inference, Api.safety, Api.memory],
        providers,
        provider_data,
        models=[
            ModelInput(
                model_id=model,
            )
            for model in inference_models
        ],
        shields=[safety_shield] if safety_shield else [],
    )
    return test_stack
