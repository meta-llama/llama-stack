# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict

from llama_stack.distribution.datatypes import Api, ProviderSpec

from .config import MetaReferenceImplConfig


async def get_provider_impl(
    config: MetaReferenceImplConfig, deps: Dict[Api, ProviderSpec]
):
    from .agents import MetaReferenceAgentsImpl

    assert isinstance(
        config, MetaReferenceImplConfig
    ), f"Unexpected config type: {type(config)}"

    impl = MetaReferenceAgentsImpl(
        config,
        deps[Api.inference],
        deps[Api.memory],
        deps[Api.safety],
    )
    await impl.initialize()
    return impl
