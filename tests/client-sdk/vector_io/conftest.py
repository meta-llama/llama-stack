# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


def pytest_addoption(parser):
    parser.addoption(
        "--embedding-model",
        action="store",
        default="all-MiniLM-L6-v2",
        help="Specify the embedding model to use for testing",
    )


def pytest_generate_tests(metafunc):
    if "embedding_model" in metafunc.fixturenames:
        metafunc.parametrize(
            "embedding_model",
            [metafunc.config.getoption("--embedding-model")],
        )
