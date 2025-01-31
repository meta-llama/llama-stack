# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import os

import pytest

from llama_stack import LlamaStackAsLibraryClient
from llama_stack.providers.tests.env import get_env_or_fail
from llama_stack_client import LlamaStackClient
from report import Report


def pytest_configure(config):
    config.option.tbstyle = "short"
    config.option.disable_warnings = True
    # Note:
    # if report_path is not provided (aka no option --report in the pytest command),
    # it will be set to False
    # if --report will give None ( in this case we infer report_path)
    # if --report /a/b is provided, it will be set to the path provided
    # We want to handle all these cases and hence explicitly check for False
    report_path = config.getoption("--report")
    if report_path is not False:
        config.pluginmanager.register(Report(report_path))


TEXT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
VISION_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"


def pytest_addoption(parser):
    parser.addoption(
        "--report",
        action="store",
        default=False,
        nargs="?",
        type=str,
        help="Path where the test report should be written, e.g. --report=/path/to/report.md",
    )
    parser.addoption(
        "--inference-model",
        action="store",
        default=TEXT_MODEL,
        help="Specify the inference model to use for testing",
    )
    parser.addoption(
        "--vision-inference-model",
        action="store",
        default=VISION_MODEL,
        help="Specify the vision inference model to use for testing",
    )
    parser.addoption(
        "--safety-shield",
        action="store",
        default="meta-llama/Llama-Guard-3-1B",
        help="Specify the safety shield model to use for testing",
    )


@pytest.fixture(scope="session")
def provider_data():
    # check env for tavily secret, brave secret and inject all into provider data
    provider_data = {}
    if os.environ.get("TAVILY_SEARCH_API_KEY"):
        provider_data["tavily_search_api_key"] = os.environ["TAVILY_SEARCH_API_KEY"]
    if os.environ.get("BRAVE_SEARCH_API_KEY"):
        provider_data["brave_search_api_key"] = os.environ["BRAVE_SEARCH_API_KEY"]
    return provider_data if len(provider_data) > 0 else None


@pytest.fixture(scope="session")
def llama_stack_client(provider_data):
    if os.environ.get("LLAMA_STACK_CONFIG"):
        client = LlamaStackAsLibraryClient(
            get_env_or_fail("LLAMA_STACK_CONFIG"),
            provider_data=provider_data,
            skip_logger_removal=True,
        )
        if not client.initialize():
            raise RuntimeError("Initialization failed")

    elif os.environ.get("LLAMA_STACK_BASE_URL"):
        client = LlamaStackClient(
            base_url=get_env_or_fail("LLAMA_STACK_BASE_URL"),
            provider_data=provider_data,
        )
    else:
        raise ValueError("LLAMA_STACK_CONFIG or LLAMA_STACK_BASE_URL must be set")
    return client


def pytest_generate_tests(metafunc):
    if "text_model_id" in metafunc.fixturenames:
        metafunc.parametrize(
            "text_model_id",
            [metafunc.config.getoption("--inference-model")],
            scope="session",
        )
    if "vision_model_id" in metafunc.fixturenames:
        metafunc.parametrize(
            "vision_model_id",
            [metafunc.config.getoption("--vision-inference-model")],
            scope="session",
        )
