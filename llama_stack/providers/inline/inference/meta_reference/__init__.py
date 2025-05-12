# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.providers.datatypes import ProviderContext

from .config import MetaReferenceInferenceConfig


async def get_provider_impl(context: ProviderContext, config: MetaReferenceInferenceConfig, _deps: dict[str, Any]):
    from .inference import MetaReferenceInferenceImpl

    impl = MetaReferenceInferenceImpl(context, config)
    await impl.initialize()
    return impl
