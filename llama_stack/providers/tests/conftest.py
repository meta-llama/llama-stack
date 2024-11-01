# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from pathlib import Path

from dotenv import load_dotenv
from termcolor import colored


def pytest_configure(config):
    """Load environment variables at start of test run"""
    # Load from .env file if it exists
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    # Load any environment variables passed via --env
    env_vars = config.getoption("--env") or []
    for env_var in env_vars:
        key, value = env_var.split("=", 1)
        os.environ[key] = value


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--env", action="append", help="Set environment variables, e.g. --env KEY=value"
    )


def pytest_itemcollected(item):
    # Get all markers as a list
    filtered = ("asyncio", "parametrize")
    marks = [mark.name for mark in item.iter_markers() if mark.name not in filtered]
    if marks:
        marks = colored(",".join(marks), "yellow")
        item.name = f"{item.name}[{marks}]"
