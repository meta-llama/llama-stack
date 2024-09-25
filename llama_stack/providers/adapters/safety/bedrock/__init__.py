# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from typing import Any

from .config import BedrockShieldConfig


async def get_adapter_impl(config: BedrockShieldConfig, _deps) -> Any:
    from .bedrock import BedrockShieldAdapter

    impl = BedrockShieldAdapter(config)
    await impl.initialize()
    return impl