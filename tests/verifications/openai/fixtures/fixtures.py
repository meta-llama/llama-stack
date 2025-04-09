# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest
from openai import OpenAI


@pytest.fixture
def providers_model_mapping():
    """
    Mapping from model names used in test cases to provider's model names.
    """
    return {
        "fireworks": {
            "Llama-3.3-70B-Instruct": "accounts/fireworks/models/llama-v3p1-70b-instruct",
            "Llama-3.2-11B-Vision-Instruct": "accounts/fireworks/models/llama-v3p2-11b-vision-instruct",
            "Llama-4-Scout-17B-16E-Instruct": "accounts/fireworks/models/llama4-scout-instruct-basic",
            "Llama-4-Maverick-17B-128E-Instruct": "accounts/fireworks/models/llama4-maverick-instruct-basic",
        },
        "together": {
            "Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Llama-3.2-11B-Vision-Instruct": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            "Llama-4-Scout-17B-16E-Instruct": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "Llama-4-Maverick-17B-128E-Instruct": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        },
        "groq": {
            "Llama-3.3-70B-Instruct": "llama-3.3-70b-versatile",
            "Llama-3.2-11B-Vision-Instruct": "llama-3.2-11b-vision-preview",
            "Llama-4-Scout-17B-16E-Instruct": "llama-4-scout-17b-16e-instruct",
            "Llama-4-Maverick-17B-128E-Instruct": "llama-4-maverick-17b-128e-instruct",
        },
        "cerebras": {
            "Llama-3.3-70B-Instruct": "llama-3.3-70b",
        },
        "openai": {
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini",
        },
    }


@pytest.fixture
def provider_metadata():
    return {
        "fireworks": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
        "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
        "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
        "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
        "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    }


@pytest.fixture
def provider(request, provider_metadata):
    provider = request.config.getoption("--provider")
    base_url = request.config.getoption("--base-url")

    if provider and base_url and provider_metadata[provider][0] != base_url:
        raise ValueError(f"Provider {provider} is not supported for base URL {base_url}")

    if not provider:
        if not base_url:
            raise ValueError("Provider and base URL are not provided")
        for provider, metadata in provider_metadata.items():
            if metadata[0] == base_url:
                provider = provider
                break

    return provider


@pytest.fixture
def base_url(request, provider, provider_metadata):
    return request.config.getoption("--base-url") or provider_metadata[provider][0]


@pytest.fixture
def api_key(request, provider, provider_metadata):
    return request.config.getoption("--api-key") or os.getenv(provider_metadata[provider][1])


@pytest.fixture
def model_mapping(provider, providers_model_mapping):
    return providers_model_mapping[provider]


@pytest.fixture
def openai_client(base_url, api_key):
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
