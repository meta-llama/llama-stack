# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import S3ImplConfig


async def get_adapter_impl(config: S3ImplConfig, _deps):
    from .s3_files import S3FilesAdapter

    impl = S3FilesAdapter(config)
    await impl.initialize()
    return impl
