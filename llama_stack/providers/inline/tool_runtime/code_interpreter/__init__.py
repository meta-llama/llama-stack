# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .code_interpreter import CodeInterpreterToolRuntimeImpl
from .config import CodeInterpreterToolConfig

__all__ = ["CodeInterpreterToolConfig", "CodeInterpreterToolRuntimeImpl"]


async def get_provider_impl(config: CodeInterpreterToolConfig, _deps):
    impl = CodeInterpreterToolRuntimeImpl(config)
    await impl.initialize()
    return impl
