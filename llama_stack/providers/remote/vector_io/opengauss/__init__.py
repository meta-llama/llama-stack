# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.datatypes import Api

from .config import OpenGaussVectorIOConfig


async def get_adapter_impl(config: OpenGaussVectorIOConfig, deps):
    from .opengauss import OpenGaussVectorIOAdapter

    files_api = deps.get(Api.files)
    impl = OpenGaussVectorIOAdapter(config, deps[Api.inference], files_api)
    await impl.initialize()
    return impl
