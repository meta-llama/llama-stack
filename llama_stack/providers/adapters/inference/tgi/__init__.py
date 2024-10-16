# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Union

from .config import (
    DellTGIImplConfig,
    InferenceAPIImplConfig,
    InferenceEndpointImplConfig,
    TGIImplConfig,
)
from .tgi import (
    DellTGIAdapter,
    InferenceAPIAdapter,
    InferenceEndpointAdapter,
    TGIAdapter,
)


async def get_adapter_impl(
    config: Union[
        InferenceAPIImplConfig,
        InferenceEndpointImplConfig,
        TGIImplConfig,
        DellTGIImplConfig,
    ],
    _deps,
):
    if isinstance(config, DellTGIImplConfig):
        impl = DellTGIAdapter()
    elif isinstance(config, TGIImplConfig):
        impl = TGIAdapter()
    elif isinstance(config, InferenceAPIImplConfig):
        impl = InferenceAPIAdapter()
    elif isinstance(config, InferenceEndpointImplConfig):
        impl = InferenceEndpointAdapter()
    else:
        raise ValueError(
            f"Invalid configuration. Expected 'TGIAdapter', 'InferenceAPIImplConfig' or 'InferenceEndpointImplConfig'. Got {type(config)}."
        )

    await impl.initialize(config)
    return impl
