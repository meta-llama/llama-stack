# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel

from llama_stack.apis.inference import Inference

from .config import LitellmConfig


async def get_adapter_impl(config: LitellmConfig, _deps) -> Inference:
    # import dynamically so the import is used only when it is needed
    from .litellm import LitellmInferenceAdapter
    assert isinstance(config, LitellmConfig), f"Unexpected config type: {type(config)}"
    adapter = LitellmInferenceAdapter(config)
    return adapter
