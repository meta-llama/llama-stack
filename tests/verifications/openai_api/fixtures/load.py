# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

import yaml


def load_test_cases(name: str):
    fixture_dir = Path(__file__).parent / "test_cases"
    yaml_path = fixture_dir / f"{name}.yaml"
    with open(yaml_path) as f:
        return yaml.safe_load(f)
