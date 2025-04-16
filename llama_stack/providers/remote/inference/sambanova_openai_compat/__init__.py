# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.inference import Inference

from .config import SambaNovaCompatConfig


async def get_adapter_impl(config: SambaNovaCompatConfig, _deps) -> Inference:
    # import dynamically so the import is used only when it is needed
    from .sambanova import SambaNovaCompatInferenceAdapter

    adapter = SambaNovaCompatInferenceAdapter(config)
    return adapter
