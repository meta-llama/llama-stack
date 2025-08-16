# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.cli.stack._show import (
    run_stack_show_command,
)
from llama_stack.core.datatypes import BuildConfig, DistributionSpec
from llama_stack.core.utils.image_types import LlamaStackImageType


def test_stack_show_basic():
    cfg = BuildConfig(
        image_type=LlamaStackImageType.CONTAINER.value,
        distribution_spec=DistributionSpec(providers={}, description=""),
    )

    run_stack_show_command(cfg)
