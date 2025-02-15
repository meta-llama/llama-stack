# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel

from .config import ModelContextProtocolConfig


class ModelContextProtocolToolProviderDataValidator(BaseModel):
    api_key: str


async def get_adapter_impl(config: ModelContextProtocolConfig, _deps):
    from .model_context_protocol import ModelContextProtocolToolRuntimeImpl

    impl = ModelContextProtocolToolRuntimeImpl(config)
    await impl.initialize()
    return impl
