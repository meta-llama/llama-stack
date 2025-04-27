
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import base64
import mimetypes
import os
from urllib.parse import urlparse, parse_qs

import pytest

# How to run this test:
#
# LLAMA_STACK_CONFIG="template-name" pytest -v tests/integration/datasets


def data_url_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as file:
        file_content = file.read()

    base64_content = base64.b64encode(file_content).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(file_path)

    data_url = f"data:{mime_type};base64,{base64_content}"

    return data_url


@pytest.mark.skip(reason="flaky. Couldn't find 'llamastack/simpleqa' on the Hugging Face Hub")
@pytest.mark.parametrize(
    "purpose, source, provider_id, limit, total_expected",
    [
        (
            "eval/messages-answer",
            {
                "type": "uri",
                "uri": "huggingface://datasets/llamastack/simpleqa?split=train",
            },
            "huggingface",
            5, # Request 5, expect more
            10, # Assume total > 5
        ),
        (
            "eval/messages-answer",
            {
                "type": "rows",
                "rows": [
                    {
                        "messages": [{"role": "user", "content": "Hello, world!"}],
                        "answer": "Hello, world!",
                    },
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "What is the capital of France?",
                            }
                        ],
                        "answer": "Paris",
                    },
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "Third message",
                            }
                        ],
                        "answer": "Third answer",
                    },
                ],
            },
            "localfs",
            2, # Request 2, expect more
            3, # Total is 3
        ),
        (
            "eval/messages-answer",
            {
                "type": "uri",
                "uri": data_url_from_file(os.path.join(os.path.dirname(__file__), "test_dataset.csv")),
            },
            "localfs",
            3, # Request 3, expect more
            5, # Total is 5
        ),
         (
            "eval/messages-answer",
            {
                "type": "uri",
                "uri": data_url_from_file(os.path.join(os.path.dirname(__file__), "test_dataset.csv")),
            },
            "localfs",
            5, # Request all 5, expect no more
            5, # Total is 5
        ),
    ],
)
def test_register_and_iterrows(llama_stack_client, purpose, source, provider_id, limit, total_expected):
    dataset = llama_stack_client.datasets.register(
        purpose=purpose,
        source=source,
    )
    assert dataset.identifier is not None
    assert dataset.provider_id == provider_id

    # Initial request
    start_index = 0
    iterrow_response = llama_stack_client.datasets.iterrows(dataset.identifier, limit=limit, start_index=start_index)
    assert len(iterrow_response.data) == min(limit, total_expected)

    # Check pagination fields
    expected_has_more = (start_index + limit) < total_expected
    assert iterrow_response.has_more == expected_has_more

    if expected_has_more:
        assert hasattr(iterrow_response, "url"), "PaginatedResponse should have a 'url' field when has_more is True"
        assert iterrow_response.url is not None, "PaginatedResponse url should not be None when has_more is True"
        # Parse the URL to check parameters
        parsed_url = urlparse(iterrow_response.url)
        query_params = parse_qs(parsed_url.query)
        assert "start_index" in query_params, "Next page URL must contain start_index"
        assert int(query_params["start_index"][0]) == start_index + limit, "Next page URL start_index is incorrect"
        assert "limit" in query_params, "Next page URL must contain limit"
        assert int(query_params["limit"][0]) == limit, "Next page URL limit is incorrect"
        assert parsed_url.path == f"/datasets/{dataset.identifier}/iterrows", "Next page URL path is incorrect"

        # Optionally, make a request to the next page URL (requires client base_url to be set)
        # This is more complex as it bypasses the client method

    else:
        assert not hasattr(iterrow_response, "url") or iterrow_response.url is None, "PaginatedResponse url should be None or missing when has_more is False"


    # List and check presence
    dataset_list = llama_stack_client.datasets.list()
    assert dataset.identifier in [d.identifier for d in dataset_list]

    # Unregister and check absence
    llama_stack_client.datasets.unregister(dataset.identifier)
    dataset_list = llama_stack_client.datasets.list()
    assert dataset.identifier not in [d.identifier for d in dataset_list]
