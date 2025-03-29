# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import TGIImplConfig


async def get_adapter_impl(config: TGIImplConfig, _deps):
    from .tgi import TGIAdapter

    if isinstance(config, TGIImplConfig):
        impl = TGIAdapter()
    else:
        raise ValueError(
            f"Invalid configuration. Expected 'TGIAdapter', 'InferenceAPIImplConfig' or 'InferenceEndpointImplConfig'. Got {type(config)}."
        )

    await impl.initialize(config)
    return impl
