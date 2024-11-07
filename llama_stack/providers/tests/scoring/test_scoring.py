# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest

# How to run this test:
#
# pytest llama_stack/providers/tests/scoring/test_scoring.py
#   -m "meta_reference"
#   -v -s --tb=short --disable-warnings


class TestScoring:
    @pytest.mark.asyncio
    async def test_scoring_functions_list(self, scoring_stack):
        # NOTE: this needs you to ensure that you are starting from a clean state
        # but so far we don't have an unregister API unfortunately, so be careful
        _, scoring_functions_impl = scoring_stack
        # response = await datasets_impl.list_datasets()
        # assert isinstance(response, list)
        # assert len(response) == 0
