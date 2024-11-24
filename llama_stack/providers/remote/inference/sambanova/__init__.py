# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel

from .config import SambanovaImplConfig


class SambanovaProviderDataValidator(BaseModel):
    sambanova_api_key: str


async def get_adapter_impl(config: SambanovaImplConfig, _deps):
    from .sambanova import SambanovaInferenceAdapter

    assert isinstance(
        config, SambanovaImplConfig
    ), f"Unexpected config type: {type(config)}"
    impl = SambanovaInferenceAdapter(config)
    await impl.initialize()
    return impl
