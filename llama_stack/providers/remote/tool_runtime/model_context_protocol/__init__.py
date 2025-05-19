# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel

from llama_stack.apis.datatypes import Api

from .config import ModelContextProtocolConfig


class ModelContextProtocolToolProviderDataValidator(BaseModel):
    api_key: str


async def get_adapter_impl(config: ModelContextProtocolConfig, deps: dict[Api, Any]):
    from .model_context_protocol import ModelContextProtocolToolRuntimeImpl

    impl = ModelContextProtocolToolRuntimeImpl(config, deps)
    await impl.initialize()
    return impl
