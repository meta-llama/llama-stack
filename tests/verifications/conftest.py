# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


def pytest_addoption(parser):
    parser.addoption(
        "--base-url",
        action="store",
        help="Base URL for OpenAI compatible API",
    )
    parser.addoption(
        "--api-key",
        action="store",
        help="API key",
    )
    parser.addoption(
        "--provider",
        action="store",
        help="Provider to use for testing",
    )


pytest_plugins = [
    "tests.verifications.openai.fixtures.fixtures",
]
