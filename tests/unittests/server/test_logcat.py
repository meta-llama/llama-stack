# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
import logging
import os
import unittest

from llama_stack import logcat


class TestLogcat(unittest.TestCase):
    def setUp(self):
        self.original_env = os.environ.get("LLAMA_STACK_LOGGING")

        self.log_output = io.StringIO()
        self._init_logcat()

    def tearDown(self):
        if self.original_env is not None:
            os.environ["LLAMA_STACK_LOGGING"] = self.original_env
        else:
            os.environ.pop("LLAMA_STACK_LOGGING", None)

    def _init_logcat(self):
        logcat.init(default_level=logging.DEBUG)
        self.handler = logging.StreamHandler(self.log_output)
        self.handler.setFormatter(logging.Formatter("[%(category)s] %(message)s"))
        logcat._logger.handlers.clear()
        logcat._logger.addHandler(self.handler)

    def test_basic_logging(self):
        logcat.info("server", "Info message")
        logcat.warning("server", "Warning message")
        logcat.error("server", "Error message")

        output = self.log_output.getvalue()
        self.assertIn("[server] Info message", output)
        self.assertIn("[server] Warning message", output)
        self.assertIn("[server] Error message", output)

    def test_different_categories(self):
        # Log messages with different categories
        logcat.info("server", "Server message")
        logcat.info("inference", "Inference message")
        logcat.info("router", "Router message")

        output = self.log_output.getvalue()
        self.assertIn("[server] Server message", output)
        self.assertIn("[inference] Inference message", output)
        self.assertIn("[router] Router message", output)

    def test_env_var_control(self):
        os.environ["LLAMA_STACK_LOGGING"] = "server=debug;inference=warning"
        self._init_logcat()

        # These should be visible based on the environment settings
        logcat.debug("server", "Server debug message")
        logcat.info("server", "Server info message")
        logcat.warning("inference", "Inference warning message")
        logcat.error("inference", "Inference error message")

        # These should be filtered out based on the environment settings
        logcat.debug("inference", "Inference debug message")
        logcat.info("inference", "Inference info message")

        output = self.log_output.getvalue()
        self.assertIn("[server] Server debug message", output)
        self.assertIn("[server] Server info message", output)
        self.assertIn("[inference] Inference warning message", output)
        self.assertIn("[inference] Inference error message", output)

        self.assertNotIn("[inference] Inference debug message", output)
        self.assertNotIn("[inference] Inference info message", output)

    def test_invalid_category(self):
        logcat.info("nonexistent", "This message should not be logged")

        # Check that the message was not logged
        output = self.log_output.getvalue()
        self.assertNotIn("[nonexistent] This message should not be logged", output)


if __name__ == "__main__":
    unittest.main()
