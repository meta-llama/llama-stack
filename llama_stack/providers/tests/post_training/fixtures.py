# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
import pytest_asyncio

from llama_stack.apis.common.type_system import *  # noqa: F403
from llama_stack.apis.common.content_types import URL
from llama_stack.apis.datasets import DatasetInput
from llama_stack.apis.models import ModelInput

from llama_stack.distribution.datatypes import Api, Provider

from llama_stack.providers.tests.resolver import construct_stack_for_test

from ..conftest import ProviderFixture


@pytest.fixture(scope="session")
def post_training_torchtune() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="torchtune",
                provider_type="inline::torchtune",
                config={},
            )
        ],
    )


POST_TRAINING_FIXTURES = ["torchtune"]


@pytest_asyncio.fixture(scope="session")
async def post_training_stack(request):
    fixture_dict = request.param

    providers = {}
    provider_data = {}
    for key in ["post_training", "datasetio"]:
        fixture = request.getfixturevalue(f"{key}_{fixture_dict[key]}")
        providers[key] = fixture.providers
        if fixture.provider_data:
            provider_data.update(fixture.provider_data)

    test_stack = await construct_stack_for_test(
        [Api.post_training, Api.datasetio],
        providers,
        provider_data,
        models=[ModelInput(model_id="meta-llama/Llama-3.2-3B-Instruct")],
        datasets=[
            DatasetInput(
                dataset_id="alpaca",
                provider_id="huggingface",
                url=URL(uri="https://huggingface.co/datasets/tatsu-lab/alpaca"),
                metadata={
                    "path": "tatsu-lab/alpaca",
                    "split": "train",
                },
                dataset_schema={
                    "instruction": StringType(),
                    "input": StringType(),
                    "output": StringType(),
                    "text": StringType(),
                },
            ),
        ],
    )

    return test_stack.impls[Api.post_training]
