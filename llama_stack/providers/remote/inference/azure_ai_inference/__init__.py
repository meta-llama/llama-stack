# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from .azure_ai_inference import AzureAIInferenceAdapter
from .config import AzureAIInferenceConfig


async def get_adapter_impl(config: AzureAIInferenceConfig, _deps):
    assert isinstance(config, AzureAIInferenceConfig), f"Unexpected config type: {type(config)}"

    impl = AzureAIInferenceAdapter(config)

    await impl.initialize()

    return impl
