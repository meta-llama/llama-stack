# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel
from termcolor import colored

from llama_stack.distribution.datatypes import Provider
from llama_stack.providers.datatypes import RemoteProviderConfig

from .env import get_env_or_fail


class ProviderFixture(BaseModel):
    providers: List[Provider]
    provider_data: Optional[Dict[str, Any]] = None


def remote_stack_fixture() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="remote",
                provider_type="remote",
                config=RemoteProviderConfig(
                    host=get_env_or_fail("REMOTE_STACK_HOST"),
                    port=int(get_env_or_fail("REMOTE_STACK_PORT")),
                ).model_dump(),
            )
        ],
    )


def pytest_configure(config):
    config.option.tbstyle = "short"
    config.option.disable_warnings = True

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
    parser.addoption(
        "--providers",
        default="",
        help=(
            "Provider configuration in format: api1=provider1,api2=provider2. "
            "Example: --providers inference=ollama,safety=meta-reference"
        ),
    )
    """Add custom command line options"""
    parser.addoption(
        "--env", action="append", help="Set environment variables, e.g. --env KEY=value"
    )


def make_provider_id(providers: Dict[str, str]) -> str:
    return ":".join(f"{api}={provider}" for api, provider in sorted(providers.items()))


def get_provider_marks(providers: Dict[str, str]) -> List[Any]:
    marks = []
    for provider in providers.values():
        marks.append(getattr(pytest.mark, provider))
    return marks


def get_provider_fixture_overrides(
    config, available_fixtures: Dict[str, List[str]]
) -> Optional[List[pytest.param]]:
    provider_str = config.getoption("--providers")
    if not provider_str:
        return None

    fixture_dict = parse_fixture_string(provider_str, available_fixtures)
    return [
        pytest.param(
            fixture_dict,
            id=make_provider_id(fixture_dict),
            marks=get_provider_marks(fixture_dict),
        )
    ]


def parse_fixture_string(
    provider_str: str, available_fixtures: Dict[str, List[str]]
) -> Dict[str, str]:
    """Parse provider string of format 'api1=provider1,api2=provider2'"""
    if not provider_str:
        return {}

    fixtures = {}
    pairs = provider_str.split(",")
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(
                f"Invalid provider specification: {pair}. Expected format: api=provider"
            )
        api, fixture = pair.split("=")
        if api not in available_fixtures:
            raise ValueError(
                f"Unknown API: {api}. Available APIs: {list(available_fixtures.keys())}"
            )
        if fixture not in available_fixtures[api]:
            raise ValueError(
                f"Unknown provider '{fixture}' for API '{api}'. "
                f"Available providers: {list(available_fixtures[api])}"
            )
        fixtures[api] = fixture

    # Check that all provided APIs are supported
    for api in available_fixtures.keys():
        if api not in fixtures:
            raise ValueError(
                f"Missing provider fixture for API '{api}'. Available providers: "
                f"{list(available_fixtures[api])}"
            )
    return fixtures


def pytest_itemcollected(item):
    # Get all markers as a list
    filtered = ("asyncio", "parametrize")
    marks = [mark.name for mark in item.iter_markers() if mark.name not in filtered]
    if marks:
        marks = colored(",".join(marks), "yellow")
        item.name = f"{item.name}[{marks}]"


pytest_plugins = [
    "llama_stack.providers.tests.inference.fixtures",
    "llama_stack.providers.tests.safety.fixtures",
    "llama_stack.providers.tests.memory.fixtures",
    "llama_stack.providers.tests.agents.fixtures",
]
