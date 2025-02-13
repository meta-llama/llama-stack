# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from termcolor import colored

from llama_stack.distribution.datatypes import Provider
from llama_stack.providers.datatypes import RemoteProviderConfig

from .env import get_env_or_fail
from .report import Report


class ProviderFixture(BaseModel):
    providers: List[Provider]
    provider_data: Optional[Dict[str, Any]] = None


class TestScenario(BaseModel):
    # provider fixtures can be either a mark or a dictionary of api -> providers
    provider_fixtures: Dict[str, str] = Field(default_factory=dict)
    fixture_combo_id: Optional[str] = None


class APITestConfig(BaseModel):
    scenarios: List[TestScenario] = Field(default_factory=list)
    inference_models: List[str] = Field(default_factory=list)

    # test name format should be <relative_path.py>::<test_name>
    tests: List[str] = Field(default_factory=list)


class MemoryApiTestConfig(APITestConfig):
    embedding_model: Optional[str] = Field(default_factory=None)


class AgentsApiTestConfig(APITestConfig):
    safety_shield: Optional[str] = Field(default_factory=None)


class TestConfig(BaseModel):
    inference: Optional[APITestConfig] = None
    agents: Optional[AgentsApiTestConfig] = None
    memory: Optional[MemoryApiTestConfig] = None


def get_test_config_from_config_file(metafunc_config):
    config_file = metafunc_config.getoption("--config")
    if config_file is None:
        return None

    config_file_path = Path(__file__).parent / config_file
    if not config_file_path.exists():
        raise ValueError(
            f"Test config {config_file} was specified but not found. Please make sure it exists in the llama_stack/providers/tests directory."
        )
    with open(config_file_path, "r") as config_file:
        config = yaml.safe_load(config_file)
        return TestConfig(**config)


def get_test_config_for_api(metafunc_config, api):
    test_config = get_test_config_from_config_file(metafunc_config)
    if test_config is None:
        return None
    return getattr(test_config, api)


def get_provider_fixture_overrides_from_test_config(metafunc_config, api, default_provider_fixture_combinations):
    api_config = get_test_config_for_api(metafunc_config, api)
    if api_config is None:
        return None

    fixture_combo_ids = set()
    custom_provider_fixture_combos = []
    for scenario in api_config.scenarios:
        if scenario.fixture_combo_id:
            fixture_combo_ids.add(scenario.fixture_combo_id)
        else:
            custom_provider_fixture_combos.append(
                pytest.param(
                    scenario.provider_fixtures,
                    id=scenario.provider_fixtures.get("inference") or "",
                )
            )

    if len(fixture_combo_ids) > 0:
        for default_fixture in default_provider_fixture_combinations:
            if default_fixture.id in fixture_combo_ids:
                custom_provider_fixture_combos.append(default_fixture)
    return custom_provider_fixture_combos


def remote_stack_fixture() -> ProviderFixture:
    if url := os.getenv("REMOTE_STACK_URL", None):
        config = RemoteProviderConfig.from_url(url)
    else:
        config = RemoteProviderConfig(
            host=get_env_or_fail("REMOTE_STACK_HOST"),
            port=int(get_env_or_fail("REMOTE_STACK_PORT")),
        )
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="test::remote",
                provider_type="test::remote",
                config=config.model_dump(),
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

    if config.getoption("--output") is not None:
        config.pluginmanager.register(Report(config.getoption("--output")))


def pytest_addoption(parser):
    parser.addoption(
        "--providers",
        default="",
        help=(
            "Provider configuration in format: api1=provider1,api2=provider2. "
            "Example: --providers inference=ollama,safety=meta-reference"
        ),
    )
    parser.addoption(
        "--config",
        action="store",
        help="Set test config file (supported format: YAML), e.g. --config=test_config.yml",
    )
    parser.addoption(
        "--output",
        action="store",
        help="Set output file for test report, e.g. --output=pytest_report.md",
    )
    """Add custom command line options"""
    parser.addoption("--env", action="append", help="Set environment variables, e.g. --env KEY=value")
    parser.addoption(
        "--inference-model",
        action="store",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Specify the inference model to use for testing",
    )
    parser.addoption(
        "--safety-shield",
        action="store",
        default="meta-llama/Llama-Guard-3-1B",
        help="Specify the safety shield to use for testing",
    )
    parser.addoption(
        "--embedding-model",
        action="store",
        default=None,
        help="Specify the embedding model to use for testing",
    )
    parser.addoption(
        "--judge-model",
        action="store",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Specify the judge model to use for testing",
    )


def make_provider_id(providers: Dict[str, str]) -> str:
    return ":".join(f"{api}={provider}" for api, provider in sorted(providers.items()))


def get_provider_marks(providers: Dict[str, str]) -> List[Any]:
    marks = []
    for provider in providers.values():
        marks.append(getattr(pytest.mark, provider))
    return marks


def get_provider_fixture_overrides(config, available_fixtures: Dict[str, List[str]]) -> Optional[List[pytest.param]]:
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


def parse_fixture_string(provider_str: str, available_fixtures: Dict[str, List[str]]) -> Dict[str, str]:
    """Parse provider string of format 'api1=provider1,api2=provider2'"""
    if not provider_str:
        return {}

    fixtures = {}
    pairs = provider_str.split(",")
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid provider specification: {pair}. Expected format: api=provider")
        api, fixture = pair.split("=")
        if api not in available_fixtures:
            raise ValueError(f"Unknown API: {api}. Available APIs: {list(available_fixtures.keys())}")
        if fixture not in available_fixtures[api]:
            raise ValueError(
                f"Unknown provider '{fixture}' for API '{api}'. Available providers: {list(available_fixtures[api])}"
            )
        fixtures[api] = fixture

    # Check that all provided APIs are supported
    for api in available_fixtures.keys():
        if api not in fixtures:
            raise ValueError(
                f"Missing provider fixture for API '{api}'. Available providers: {list(available_fixtures[api])}"
            )
    return fixtures


def pytest_itemcollected(item):
    # Get all markers as a list
    filtered = ("asyncio", "parametrize")
    marks = [mark.name for mark in item.iter_markers() if mark.name not in filtered]
    if marks:
        marks = colored(",".join(marks), "yellow")
        item.name = f"{item.name}[{marks}]"


def pytest_collection_modifyitems(session, config, items):
    test_config = get_test_config_from_config_file(config)
    if test_config is None:
        return

    required_tests = defaultdict(set)
    for api_test_config in [
        test_config.inference,
        test_config.memory,
        test_config.agents,
    ]:
        if api_test_config is None:
            continue
        for test in api_test_config.tests:
            arr = test.split("::")
            if len(arr) != 2:
                raise ValueError(f"Invalid format for test name {test}")
            test_path, func_name = arr
            required_tests[Path(__file__).parent / test_path].add(func_name)

    new_items, deselected_items = [], []
    for item in items:
        func_name = getattr(item, "originalname", item.name)
        if func_name in required_tests[item.fspath]:
            new_items.append(item)
            continue
        deselected_items.append(item)

    items[:] = new_items
    config.hook.pytest_deselected(items=deselected_items)


pytest_plugins = [
    "llama_stack.providers.tests.inference.fixtures",
    "llama_stack.providers.tests.safety.fixtures",
    "llama_stack.providers.tests.vector_io.fixtures",
    "llama_stack.providers.tests.agents.fixtures",
    "llama_stack.providers.tests.datasetio.fixtures",
    "llama_stack.providers.tests.scoring.fixtures",
    "llama_stack.providers.tests.eval.fixtures",
    "llama_stack.providers.tests.post_training.fixtures",
    "llama_stack.providers.tests.tools.fixtures",
]
