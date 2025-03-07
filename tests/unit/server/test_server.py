# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Run the test with the following command:
# uv run pytest -v tests/unit/test_server.py

import argparse

import pytest

from llama_stack.distribution.server.server import return_flag_value


@pytest.mark.parametrize(
    "flag, env_var, env_value, args, config_file_value, default_args, expected",
    [
        # test environment variable passed
        (
            "flag1",
            "ENV_FLAG1",
            "env_value1",
            argparse.Namespace(flag1="cli_value1"),
            "config_value1",
            argparse.Namespace(flag1="default_value1"),
            "env_value1",
        ),
        # test flag passed in cli with --flag1
        (
            "flag2",
            "ENV_FLAG2",
            None,
            argparse.Namespace(flag2="cli_value2"),
            "config_value2",
            argparse.Namespace(flag2="default_value2"),
            "cli_value2",
        ),
        # test flag passed in config file - no env or cli
        (
            "flag3",
            "ENV_FLAG3",
            None,
            argparse.Namespace(),
            "config_value3",
            argparse.Namespace(flag3="default_value3"),
            "config_value3",
        ),
        # test - no env - no cli - no config file - should return default
        (
            "flag4",
            "ENV_FLAG4",
            None,
            argparse.Namespace(),
            None,
            argparse.Namespace(flag4="default_value4"),
            "default_value4",
        ),
    ],
)
def test_return_flag_value(flag, env_var, env_value, args, config_file_value, default_args, expected, monkeypatch):
    if env_value is not None:
        monkeypatch.setenv(env_var, env_value)
    else:
        monkeypatch.delenv(env_var, raising=False)

    assert return_flag_value(flag, env_var, args, config_file_value, default_args) == expected
