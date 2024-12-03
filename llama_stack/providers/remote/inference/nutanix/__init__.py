# Copyright (c) Nutanix, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import NutanixImplConfig


async def get_adapter_impl(config: NutanixImplConfig, _deps):
    from .nutanix import NutanixInferenceAdapter

    assert isinstance(
        config, NutanixImplConfig
    ), f"Unexpected config type: {type(config)}"
    impl = NutanixInferenceAdapter(config)
    await impl.initialize()
    return impl
