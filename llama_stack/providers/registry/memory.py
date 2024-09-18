# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.distribution.datatypes import *  # noqa: F403

EMBEDDING_DEPS = [
    "blobfile",
    "chardet",
    "pypdf",
    "sentence-transformers",
]


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.memory,
            provider_id="meta-reference",
            pip_packages=EMBEDDING_DEPS + ["faiss-cpu"],
            module="llama_stack.providers.impls.meta_reference.memory",
            config_class="llama_stack.providers.impls.meta_reference.memory.FaissImplConfig",
        ),
        remote_provider_spec(
            Api.memory,
            AdapterSpec(
                adapter_id="chromadb",
                pip_packages=EMBEDDING_DEPS + ["chromadb-client"],
                module="llama_stack.providers.adapters.memory.chroma",
            ),
        ),
        remote_provider_spec(
            Api.memory,
            AdapterSpec(
                adapter_id="pgvector",
                pip_packages=EMBEDDING_DEPS + ["psycopg2-binary"],
                module="llama_stack.providers.adapters.memory.pgvector",
                config_class="llama_stack.providers.adapters.memory.pgvector.PGVectorConfig",
            ),
        ),
    ]
