# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_toolchain.distribution.datatypes import Adapter, Api, SourceAdapter


def available_safety_adapters() -> List[Adapter]:
    return [
        SourceAdapter(
            api=Api.safety,
            adapter_id="meta-reference",
            pip_packages=[
                "codeshield",
                "torch",
                "transformers",
            ],
            module="llama_toolchain.safety.safety",
            config_class="llama_toolchain.safety.config.SafetyConfig",
        ),
    ]
