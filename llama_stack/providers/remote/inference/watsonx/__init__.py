# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.inference import Inference

from .config import WatsonXConfig


async def get_adapter_impl(config: WatsonXConfig, _deps) -> Inference:
    # import dynamically so `llama stack build` does not fail due to missing dependencies
    from .watsonx import WatsonXInferenceAdapter

    if not isinstance(config, WatsonXConfig):
        raise RuntimeError(f"Unexpected config type: {type(config)}")
    adapter = WatsonXInferenceAdapter(config)
    return adapter


__all__ = ["get_adapter_impl", "WatsonXConfig"]
