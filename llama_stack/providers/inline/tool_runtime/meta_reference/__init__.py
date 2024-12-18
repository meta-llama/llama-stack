# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from pydantic import BaseModel

from .config import MetaReferenceToolRuntimeConfig
from .meta_reference import MetaReferenceToolRuntimeImpl


class MetaReferenceProviderDataValidator(BaseModel):
    api_key: str


async def get_provider_impl(
    config: MetaReferenceToolRuntimeConfig, _deps: Dict[str, Any]
):
    impl = MetaReferenceToolRuntimeImpl(config)
    await impl.initialize()
    return impl
