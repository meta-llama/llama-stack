# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest
from llama_stack.apis.common.type_system import *  # noqa: F403
from llama_stack.apis.datasetio import *  # noqa: F403
from llama_stack.distribution.datatypes import *  # noqa: F403
import base64
import mimetypes
from pathlib import Path

# How to run this test:
#
# pytest llama_stack/providers/tests/datasetio/test_datasetio.py
#   -m "meta_reference"
#   -v -s --tb=short --disable-warnings


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
    datasets_impl: Datasets,
    for_generation=False,
    for_rag=False,
    dataset_id="test_dataset",
):
    if for_rag:
        test_file = Path(os.path.abspath(__file__)).parent / "test_rag_dataset.csv"
    else:
        test_file = Path(os.path.abspath(__file__)).parent / "test_dataset.csv"
    test_url = data_url_from_file(str(test_file))

    if for_generation:
        dataset_schema = {
            "expected_answer": StringType(),
            "input_query": StringType(),
            "chat_completion_input": ChatCompletionInputType(),
        }
    elif for_rag:
        dataset_schema = {
            "expected_answer": StringType(),
            "input_query": StringType(),
            "generated_answer": StringType(),
            "context": StringType(),
        }
    else:
        dataset_schema = {
            "expected_answer": StringType(),
            "input_query": StringType(),
            "generated_answer": StringType(),
        }

    await datasets_impl.register_dataset(
        dataset_id=dataset_id,
        dataset_schema=dataset_schema,
        url=URL(uri=test_url),
    )


class TestDatasetIO:
    @pytest.mark.asyncio
    async def test_datasets_list(self, datasetio_stack):
        # NOTE: this needs you to ensure that you are starting from a clean state
        # but so far we don't have an unregister API unfortunately, so be careful
        _, datasets_impl = datasetio_stack
        response = await datasets_impl.list_datasets()
        assert isinstance(response, list)
        assert len(response) == 0

    @pytest.mark.asyncio
    async def test_register_dataset(self, datasetio_stack):
        _, datasets_impl = datasetio_stack
        await register_dataset(datasets_impl)
        response = await datasets_impl.list_datasets()
        assert isinstance(response, list)
        assert len(response) == 1
        assert response[0].identifier == "test_dataset"

        with pytest.raises(Exception) as exc_info:
            # unregister a dataset that does not exist
            await datasets_impl.unregister_dataset("test_dataset2")

        await datasets_impl.unregister_dataset("test_dataset")
        response = await datasets_impl.list_datasets()
        assert isinstance(response, list)
        assert len(response) == 0

        with pytest.raises(Exception) as exc_info:
            await datasets_impl.unregister_dataset("test_dataset")

    @pytest.mark.asyncio
    async def test_get_rows_paginated(self, datasetio_stack):
        datasetio_impl, datasets_impl = datasetio_stack
        await register_dataset(datasets_impl)
        response = await datasetio_impl.get_rows_paginated(
            dataset_id="test_dataset",
            rows_in_page=3,
        )
        assert isinstance(response.rows, list)
        assert len(response.rows) == 3
        assert response.next_page_token == "3"

        provider = datasetio_impl.routing_table.get_provider_impl("test_dataset")
        if provider.__provider_spec__.provider_type == "remote":
            pytest.skip("remote provider doesn't support get_rows_paginated")

        # iterate over all rows
        response = await datasetio_impl.get_rows_paginated(
            dataset_id="test_dataset",
            rows_in_page=2,
            page_token=response.next_page_token,
        )
        assert isinstance(response.rows, list)
        assert len(response.rows) == 2
        assert response.next_page_token == "5"
