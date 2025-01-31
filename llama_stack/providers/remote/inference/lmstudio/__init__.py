# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.inference import Inference

from .config import LMSTUDIOConfig


async def get_adapter_impl(config: LMSTUDIOConfig, _deps) -> Inference:
    # import dynamically so `llama stack build` does not fail due to missing dependencies
    from .lmstudio import LMSTUDIOInferenceAdapter

    if not isinstance(config, LMSTUDIOConfig):
        raise RuntimeError(f"Unexpected config type: {type(config)}")
    adapter = LMSTUDIOInferenceAdapter(config)
    return adapter


__all__ = ["get_adapter_impl", "LMSTUDIOConfig"]
