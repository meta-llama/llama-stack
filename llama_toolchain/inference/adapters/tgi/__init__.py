# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import TGIImplConfig
from .tgi import InferenceEndpointAdapter, LocalTGIAdapter


async def get_adapter_impl(config: TGIImplConfig, _deps):
    assert isinstance(config, TGIImplConfig), f"Unexpected config type: {type(config)}"

    if config.is_local_tgi():
        impl = LocalTGIAdapter(config)
    elif config.is_inference_endpoint():
        impl = InferenceEndpointAdapter(config)
    else:
        raise ValueError(
            "Invalid configuration. Specify either a local URL or Inference Endpoint details."
        )

    await impl.initialize()
    return impl
