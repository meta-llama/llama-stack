# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from pydantic import BaseModel

from .config import AnthropicConfig


class AnthropicProviderDataValidator(BaseModel):
    anthropic_api_key: Optional[str] = None


async def get_adapter_impl(config: AnthropicConfig, _deps):
    from .anthropic import AnthropicInferenceAdapter

    impl = AnthropicInferenceAdapter(config)
    await impl.initialize()
    return impl
