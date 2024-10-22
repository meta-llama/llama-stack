# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from llama_stack.apis.inference import Inference
from pytest_httpx import HTTPXMock

pytestmark = pytest.mark.asyncio


async def test_chat_completion(
    mock_health: HTTPXMock,
    mock_chat_completion: HTTPXMock,
    client: Inference,
    base_url: str,
) -> None:
    """
    Test that health endpoints are checked when chat_completion is called.
    """
    client = await client

    await client.chat_completion(
        model="Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": "BOGUS"}],
        stream=False,
    )


# TODO(mf): test stream=True for each case
# TODO(mf): test completion
# TODO(mf): test embedding
