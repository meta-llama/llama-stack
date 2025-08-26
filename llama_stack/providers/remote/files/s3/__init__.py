# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.core.datatypes import Api

from .config import S3FilesImplConfig


async def get_adapter_impl(config: S3FilesImplConfig, deps: dict[Api, Any]):
    from .files import S3FilesImpl

    # TODO: authorization policies and user separation
    impl = S3FilesImpl(config)
    await impl.initialize()
    return impl
