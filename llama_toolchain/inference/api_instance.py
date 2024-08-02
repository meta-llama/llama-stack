# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# from .api.config import ImplType, InferenceConfig


# async def get_inference_api_instance(config: InferenceConfig):
#     if config.impl_config.impl_type == ImplType.inline.value:
#         from .inference import InferenceImpl

#         return InferenceImpl(config.impl_config)
#     elif config.impl_config.impl_type == ImplType.ollama.value:
#         from .ollama import OllamaInference

#         return OllamaInference(config.impl_config)

#     from .client import InferenceClient

#     return InferenceClient(config.impl_config.url)
