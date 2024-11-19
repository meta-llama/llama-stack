# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ._config import NVIDIAConfig
from ._nvidia import NVIDIAInferenceAdapter


async def get_adapter_impl(config: NVIDIAConfig, _deps) -> NVIDIAInferenceAdapter:
    if not isinstance(config, NVIDIAConfig):
        raise RuntimeError(f"Unexpected config type: {type(config)}")
    adapter = NVIDIAInferenceAdapter(config)
    return adapter


__all__ = ["get_adapter_impl", "NVIDIAConfig"]
