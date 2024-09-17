# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.distribution.datatypes import *  # noqa: F403


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.inference,
            provider_id="meta-reference",
            pip_packages=[
                "accelerate",
                "blobfile",
                "fairscale",
                "fbgemm-gpu==0.8.0",
                "torch",
                "transformers",
                "zmq",
            ],
            module="llama_stack.providers.impls.meta_reference.inference",
            config_class="llama_stack.providers.impls.meta_reference.inference.MetaReferenceImplConfig",
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_id="ollama",
                pip_packages=["ollama"],
                module="llama_stack.providers.adapters.inference.ollama",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_id="tgi",
                pip_packages=["huggingface_hub"],
                module="llama_stack.providers.adapters.inference.tgi",
                config_class="llama_stack.providers.adapters.inference.tgi.TGIImplConfig",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_id="fireworks",
                pip_packages=[
                    "fireworks-ai",
                ],
                module="llama_stack.providers.adapters.inference.fireworks",
                config_class="llama_stack.providers.adapters.inference.fireworks.FireworksImplConfig",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_id="together",
                pip_packages=[
                    "together",
                ],
                module="llama_stack.providers.adapters.inference.together",
                config_class="llama_stack.providers.adapters.inference.together.TogetherImplConfig",
            ),
        ),
    ]
