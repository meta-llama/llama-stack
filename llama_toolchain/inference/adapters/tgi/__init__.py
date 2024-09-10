# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import TGIImplConfig
from .tgi import InferenceEndpointAdapter, TGIAdapter


async def get_adapter_impl(config: TGIImplConfig, _deps):
    assert isinstance(config, TGIImplConfig), f"Unexpected config type: {type(config)}"

    if config.url is not None:
        impl = TGIAdapter(config)
    elif config.is_inference_endpoint():
        impl = InferenceEndpointAdapter(config)
    else:
        raise ValueError(
            "Invalid configuration. Specify either an URL or HF Inference Endpoint details (namespace and endpoint name)."
        )

    await impl.initialize()
    return impl
