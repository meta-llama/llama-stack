# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import pathlib


class TestCase:
    _apis = ["chat_completion", "completion"]
    _jsonblob = {}

    def __init__(self, name):
        # loading all test cases
        if self._jsonblob == {}:
            for api in self._apis:
                with open(pathlib.Path(__file__).parent / f"{api}.json", "r") as f:
                    TestCase._jsonblob.update({f"{api}-{k}": v for k, v in json.load(f).items()})

        # loading this test case
        tc = self._jsonblob.get(name)
        if tc is None:
            raise ValueError(f"Test case {name} not found")

        # these are the only fields we need
        self.name = tc.get("name")
        self.data = tc.get("data")

    def __getitem__(self, key):
        return self.data[key]
