# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest

from llama_stack.apis.inference import Inference
from llama_stack.providers.adapters.inference.nvidia import (
    get_adapter_impl,
    NVIDIAConfig,
)


def pytest_collection_modifyitems(config, items):
    """
    Skip all integration tests if NVIDIA_API_KEY is not set and --base-url
    includes "https://integrate.api.nvidia.com". It is needed to access the
    hosted preview APIs.
    """
    if "integrate.api.nvidia.com" in config.getoption(
        "--base-url"
    ) and not os.environ.get("NVIDIA_API_KEY"):
        skip_nvidia = pytest.mark.skip(
            reason="NVIDIA_API_KEY environment variable must be set to access integrate.api.nvidia.com"
        )
        for item in items:
            item.add_marker(skip_nvidia)


def pytest_addoption(parser):
    parser.addoption(
        "--base-url",
        action="store",
        default="http://localhost:8000",
        help="Base URL for the tests",
    )
    parser.addoption(
        "--model",
        action="store",
        default="Llama-3-8B-Instruct",
        help="Model option for the tests",
    )


@pytest.fixture
def base_url(request):
    return request.config.getoption("--base-url")


@pytest.fixture
def model(request):
    return request.config.getoption("--model")


@pytest.fixture
def client(base_url: str) -> Inference:
    return get_adapter_impl(
        NVIDIAConfig(
            base_url=base_url,
            api_key=os.environ.get("NVIDIA_API_KEY"),
        ),
        {},
    )
