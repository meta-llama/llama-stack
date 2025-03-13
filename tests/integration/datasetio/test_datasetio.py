# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import mimetypes
import os
from pathlib import Path

import pytest

# How to run this test:
#
# LLAMA_STACK_CONFIG="template-name" pytest -v tests/integration/datasetio


@pytest.fixture
def dataset_for_test(llama_stack_client):
    dataset_id = "test_dataset"
    register_dataset(llama_stack_client, dataset_id=dataset_id)
    yield
    # Teardown - this always runs, even if the test fails
    try:
        llama_stack_client.datasets.unregister(dataset_id)
    except Exception as e:
        print(f"Warning: Failed to unregister test_dataset: {e}")


def data_url_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as file:
        file_content = file.read()

    base64_content = base64.b64encode(file_content).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(file_path)

    data_url = f"data:{mime_type};base64,{base64_content}"

    return data_url


def register_dataset(llama_stack_client, for_generation=False, for_rag=False, dataset_id="test_dataset"):
    if for_rag:
        test_file = Path(os.path.abspath(__file__)).parent / "test_rag_dataset.csv"
    else:
        test_file = Path(os.path.abspath(__file__)).parent / "test_dataset.csv"
    test_url = data_url_from_file(str(test_file))

    if for_generation:
        dataset_schema = {
            "expected_answer": {"type": "string"},
            "input_query": {"type": "string"},
            "chat_completion_input": {"type": "chat_completion_input"},
        }
    elif for_rag:
        dataset_schema = {
            "expected_answer": {"type": "string"},
            "input_query": {"type": "string"},
            "generated_answer": {"type": "string"},
            "context": {"type": "string"},
        }
    else:
        dataset_schema = {
            "expected_answer": {"type": "string"},
            "input_query": {"type": "string"},
            "generated_answer": {"type": "string"},
        }

    dataset_providers = [x for x in llama_stack_client.providers.list() if x.api == "datasetio"]
    dataset_provider_id = dataset_providers[0].provider_id

    llama_stack_client.datasets.register(
        dataset_id=dataset_id,
        dataset_schema=dataset_schema,
        url=dict(uri=test_url),
        provider_id=dataset_provider_id,
    )


def test_register_unregister_dataset(llama_stack_client):
    register_dataset(llama_stack_client)
    response = llama_stack_client.datasets.list()
    assert isinstance(response, list)
    assert len(response) == 1
    assert response[0].identifier == "test_dataset"

    llama_stack_client.datasets.unregister("test_dataset")
    response = llama_stack_client.datasets.list()
    assert isinstance(response, list)
    assert len(response) == 0


def test_get_rows_paginated(llama_stack_client, dataset_for_test):
    response = llama_stack_client.datasetio.get_rows_paginated(
        dataset_id="test_dataset",
        rows_in_page=3,
    )
    assert isinstance(response.rows, list)
    assert len(response.rows) == 3
    assert response.next_page_token == "3"

    # iterate over all rows
    response = llama_stack_client.datasetio.get_rows_paginated(
        dataset_id="test_dataset",
        rows_in_page=2,
        page_token=response.next_page_token,
    )
    assert isinstance(response.rows, list)
    assert len(response.rows) == 2
    assert response.next_page_token == "5"
