# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import yaml
from pydantic import BaseModel, Field


@dataclass
class APITestConfig(BaseModel):

    class Fixtures(BaseModel):
        # provider fixtures can be either a mark or a dictionary of api -> providers
        provider_fixtures: List[Dict[str, str]]
        inference_models: List[str] = Field(default_factory=list)
        safety_shield: Optional[str]
        embedding_model: Optional[str]

    fixtures: Fixtures
    tests: List[str] = Field(default_factory=list)

    # test name format should be <relative_path.py>::<test_name>


class TestConfig(BaseModel):

    inference: APITestConfig
    agent: APITestConfig
    memory: APITestConfig


CONFIG_CACHE = None


def try_load_config_file_cached(config_file):
    if config_file is None:
        return None
    if CONFIG_CACHE is not None:
        return CONFIG_CACHE

    config_file_path = Path(__file__).parent / config_file
    if not config_file_path.exists():
        raise ValueError(
            f"Test config {config_file} was specified but not found. Please make sure it exists in the llama_stack/providers/tests directory."
        )
    with open(config_file_path, "r") as config_file:
        config = yaml.safe_load(config_file)
        return TestConfig(**config)


def get_provider_fixtures_from_config(
    provider_fixtures_config, default_fixture_combination
):
    custom_fixtures = []
    selected_default_param_id = set()
    for fixture_config in provider_fixtures_config:
        if "default_fixture_param_id" in fixture_config:
            selected_default_param_id.add(fixture_config["default_fixture_param_id"])
        else:
            custom_fixtures.append(
                pytest.param(fixture_config, id=fixture_config.get("inference") or "")
            )

    if len(selected_default_param_id) > 0:
        for default_fixture in default_fixture_combination:
            if default_fixture.id in selected_default_param_id:
                custom_fixtures.append(default_fixture)

    return custom_fixtures
