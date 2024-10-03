# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import DatabricksImplConfig
from .vllm import InferenceEndpointAdapter, VLLMAdapter


async def get_adapter_impl(config: DatabricksImplConfig, _deps):
    assert isinstance(config, DatabricksImplConfig), f"Unexpected config type: {type(config)}"

    if config.url is not None:
        impl = VLLMAdapter(config)
    elif config.is_inference_endpoint():
        impl = InferenceEndpointAdapter(config)
    else:
        raise ValueError(
            "Invalid configuration. Specify either an URL or HF Inference Endpoint details (namespace and endpoint name)."
        )

    await impl.initialize()
    return impl