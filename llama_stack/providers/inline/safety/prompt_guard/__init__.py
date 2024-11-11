# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import PromptGuardConfig  # noqa: F401


async def get_provider_impl(config: PromptGuardConfig, deps):
    from .prompt_guard import PromptGuardSafetyImpl

    impl = PromptGuardSafetyImpl(config, deps)
    await impl.initialize()
    return impl
