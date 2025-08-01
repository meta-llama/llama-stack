# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

from llama_stack.cli.stack._sync import (
    _run_stack_sync_command_from_build_config,
)
from llama_stack.core.datatypes import BuildConfig, DistributionSpec
from llama_stack.core.utils.image_types import LlamaStackImageType


def test_stack_sync_basic():
    # Set UV_SYSTEM_PYTHON to make it use system packages
    os.environ["UV_SYSTEM_PYTHON"] = "1"

    cfg = BuildConfig(
        image_type=LlamaStackImageType.CONTAINER.value,
        distribution_spec=DistributionSpec(providers={}, description=""),
    )

    run_config = _run_stack_sync_command_from_build_config(cfg)

    assert run_config is not None
