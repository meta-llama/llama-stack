# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest

# Skip all tests in this directory when running in GitHub Actions
in_github_actions = os.environ.get("GITHUB_ACTIONS") == "true"
if in_github_actions:
    pytest.skip("Skipping NVIDIA tests in GitHub Actions environment", allow_module_level=True)
