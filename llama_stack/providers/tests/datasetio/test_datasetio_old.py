# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import os

import pytest
import pytest_asyncio

from llama_stack.apis.common.type_system import *  # noqa: F403
from llama_stack.apis.datasetio import *  # noqa: F403
from llama_stack.distribution.datatypes import *  # noqa: F403
import base64
import mimetypes
from pathlib import Path

from llama_stack.providers.tests.resolver import resolve_impls_for_test

# How to run this test:
#
# 1. Ensure you have a conda with the right dependencies installed. This is a bit tricky
#    since it depends on the provider you are testing. On top of that you need
#    `pytest` and `pytest-asyncio` installed.
#
# 2. Copy and modify the provider_config_example.yaml depending on the provider you are testing.
#
# 3. Run:
#
# ```bash
# PROVIDER_ID=<your_provider> \
#   PROVIDER_CONFIG=provider_config.yaml \
#   pytest -s llama_stack/providers/tests/datasetio/test_datasetio.py \
#   --tb=short --disable-warnings
# ```


@pytest_asyncio.fixture(scope="session")
async def datasetio_settings():
    impls = await resolve_impls_for_test(
        Api.datasetio,
    )
    return {
        "datasetio_impl": impls[Api.datasetio],
        "datasets_impl": impls[Api.datasets],
    }


def data_url_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as file:
        file_content = file.read()

    base64_content = base64.b64encode(file_content).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(file_path)

    data_url = f"data:{mime_type};base64,{base64_content}"

    return data_url


async def register_dataset(
    datasets_impl: Datasets, for_generation=False, dataset_id="test_dataset"
):
    test_file = Path(os.path.abspath(__file__)).parent / "test_dataset.csv"
    test_url = data_url_from_file(str(test_file))

    if for_generation:
        dataset_schema = {
            "expected_answer": StringType(),
            "input_query": StringType(),
            "chat_completion_input": ChatCompletionInputType(),
        }
    else:
        dataset_schema = {
            "expected_answer": StringType(),
            "input_query": StringType(),
            "generated_answer": StringType(),
        }

    dataset = DatasetDefWithProvider(
        identifier=dataset_id,
        provider_id=os.environ.get("DATASETIO_PROVIDER_ID", None)
        or os.environ["PROVIDER_ID"],
        url=URL(
            uri=test_url,
        ),
        dataset_schema=dataset_schema,
    )
    await datasets_impl.register_dataset(dataset)


@pytest.mark.asyncio
async def test_datasets_list(datasetio_settings):
    # NOTE: this needs you to ensure that you are starting from a clean state
    # but so far we don't have an unregister API unfortunately, so be careful
    datasets_impl = datasetio_settings["datasets_impl"]
    response = await datasets_impl.list_datasets()
    assert isinstance(response, list)
    assert len(response) == 0


@pytest.mark.asyncio
async def test_datasets_register(datasetio_settings):
    # NOTE: this needs you to ensure that you are starting from a clean state
    # but so far we don't have an unregister API unfortunately, so be careful
    datasets_impl = datasetio_settings["datasets_impl"]
    await register_dataset(datasets_impl)

    response = await datasets_impl.list_datasets()
    assert isinstance(response, list)
    assert len(response) == 1

    # register same dataset with same id again will fail
    await register_dataset(datasets_impl)
    response = await datasets_impl.list_datasets()
    assert isinstance(response, list)
    assert len(response) == 1
    assert response[0].identifier == "test_dataset"


@pytest.mark.asyncio
async def test_get_rows_paginated(datasetio_settings):
    datasetio_impl = datasetio_settings["datasetio_impl"]
    datasets_impl = datasetio_settings["datasets_impl"]
    await register_dataset(datasets_impl)

    response = await datasetio_impl.get_rows_paginated(
        dataset_id="test_dataset",
        rows_in_page=3,
    )

    assert isinstance(response.rows, list)
    assert len(response.rows) == 3
    assert response.next_page_token == "3"

    # iterate over all rows
    response = await datasetio_impl.get_rows_paginated(
        dataset_id="test_dataset",
        rows_in_page=2,
        page_token=response.next_page_token,
    )

    assert isinstance(response.rows, list)
    assert len(response.rows) == 2
    assert response.next_page_token == "5"
