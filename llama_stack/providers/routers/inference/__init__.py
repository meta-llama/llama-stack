# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, List, Tuple

from llama_stack.distribution.datatypes import Api


async def get_router_impl(models_api: Api):
    from .inference import InferenceRouterImpl

    impl = InferenceRouterImpl(models_api)
    await impl.initialize()
    return impl
