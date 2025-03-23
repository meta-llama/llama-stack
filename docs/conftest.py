# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import time


def pytest_collection_modifyitems(items):
    for item in items:
        item.name = item.name.replace(' ', '_') 


def pytest_runtest_teardown(item):
    interval_seconds = os.getenv("LLAMA_STACK_TEST_INTERVAL_SECONDS")
    if interval_seconds:
        time.sleep(float(interval_seconds))


def pytest_configure(config):
    config.option.tbstyle = "short"
    config.option.disable_warnings = True
