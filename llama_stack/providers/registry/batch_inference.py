# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.providers.datatypes import (
    Api,
    InlineProviderSpec,
    ProviderSpec,
)

META_REFERENCE_DEPS = [
    "accelerate",
    "blobfile",
    "fairscale",
    "torch",
    "torchvision",
    "transformers",
    "zmq",
    "lm-format-enforcer",
    "sentence-transformers",
    "torchao==0.5.0",
    "fbgemm-gpu-genai==1.1.2",
]


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.inference,
            provider_type="inline::meta-reference",
            pip_packages=META_REFERENCE_DEPS,
            module="llama_stack.providers.inline.batch_inference.meta_reference",
            config_class="llama_stack.providers.inline.batch_inference.meta_reference.MetaReferenceInferenceConfig",
        ),
    ]
