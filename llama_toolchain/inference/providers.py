# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_toolchain.distribution.datatypes import Api, InlineProviderSpec, ProviderSpec


def available_inference_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.inference,
            provider_id="meta-reference",
            pip_packages=[
                "torch",
                "zmq",
            ],
            module="llama_toolchain.inference.inference",
            config_class="llama_toolchain.inference.inference.MetaReferenceImplConfig",
        ),
        InlineProviderSpec(
            api=Api.inference,
            provider_id="meta-ollama",
            pip_packages=[
                "ollama",
            ],
            module="llama_toolchain.inference.ollama",
            config_class="llama_toolchain.inference.ollama.OllamaImplConfig",
        ),
    ]
