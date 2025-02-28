# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import unittest

from llama_stack.distribution.stack import replace_env_vars


class TestReplaceEnvVars(unittest.TestCase):
    def setUp(self):
        # Clear any existing environment variables we'll use in tests
        for var in ["TEST_VAR", "EMPTY_VAR", "ZERO_VAR"]:
            if var in os.environ:
                del os.environ[var]

        # Set up test environment variables
        os.environ["TEST_VAR"] = "test_value"
        os.environ["EMPTY_VAR"] = ""
        os.environ["ZERO_VAR"] = "0"

    def test_simple_replacement(self):
        self.assertEqual(replace_env_vars("${env.TEST_VAR}"), "test_value")

    def test_default_value_when_not_set(self):
        self.assertEqual(replace_env_vars("${env.NOT_SET:default}"), "default")

    def test_default_value_when_set(self):
        self.assertEqual(replace_env_vars("${env.TEST_VAR:default}"), "test_value")

    def test_default_value_when_empty(self):
        self.assertEqual(replace_env_vars("${env.EMPTY_VAR:default}"), "default")

    def test_conditional_value_when_set(self):
        self.assertEqual(replace_env_vars("${env.TEST_VAR+conditional}"), "conditional")

    def test_conditional_value_when_not_set(self):
        self.assertEqual(replace_env_vars("${env.NOT_SET+conditional}"), "")

    def test_conditional_value_when_empty(self):
        self.assertEqual(replace_env_vars("${env.EMPTY_VAR+conditional}"), "")

    def test_conditional_value_with_zero(self):
        self.assertEqual(replace_env_vars("${env.ZERO_VAR+conditional}"), "conditional")

    def test_mixed_syntax(self):
        self.assertEqual(replace_env_vars("${env.TEST_VAR:default} and ${env.NOT_SET+conditional}"), "test_value and ")
        self.assertEqual(
            replace_env_vars("${env.NOT_SET:default} and ${env.TEST_VAR+conditional}"), "default and conditional"
        )

    def test_nested_structures(self):
        data = {
            "key1": "${env.TEST_VAR:default}",
            "key2": ["${env.NOT_SET:default}", "${env.TEST_VAR+conditional}"],
            "key3": {"nested": "${env.NOT_SET+conditional}"},
        }
        expected = {"key1": "test_value", "key2": ["default", "conditional"], "key3": {"nested": ""}}
        self.assertEqual(replace_env_vars(data), expected)


if __name__ == "__main__":
    unittest.main()
