# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import CodeShieldConfig


async def get_provider_impl(config: CodeShieldConfig, deps):
    from .code_scanner import MetaReferenceCodeScannerSafetyImpl

    impl = MetaReferenceCodeScannerSafetyImpl(config, deps)
    await impl.initialize()
    return impl
