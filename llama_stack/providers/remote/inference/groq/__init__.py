# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel

from .config import GroqImplConfig


class GroqProviderDataValidator(BaseModel):
    groq_api_key: str


async def get_adapter_impl(config: GroqImplConfig, _deps):
    from .groq import GroqInferenceAdapter

    assert isinstance(
        config, GroqImplConfig
    ), f"Unexpected config type: {type(config)}"
    impl = GroqInferenceAdapter(config)
    await impl.initialize()
    return impl
