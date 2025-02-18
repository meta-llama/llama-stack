# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from llama_stack.apis.inference import HealthResponse

# How to run this test:
# pytest -v -s llama_stack/providers/tests/inference/test_health.py


class TestHeatlh:
    @pytest.mark.asyncio
    async def test_health(self, inference_stack):
        inference_impl, _ = inference_stack
        response = await inference_impl.health()
        for key in response:
            assert isinstance(response[key], HealthResponse)
            assert response[key].health["status"] == "OK", response
