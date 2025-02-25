# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from pydantic import BaseModel

from .config import GeminiConfig


class GeminiProviderDataValidator(BaseModel):
    gemini_api_key: Optional[str] = None


async def get_adapter_impl(config: GeminiConfig, _deps):
    from .gemini import GeminiInferenceAdapter

    impl = GeminiInferenceAdapter(config)
    await impl.initialize()
    return impl
