# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_toolchain.core.datatypes import *  # noqa: F403

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
            provider_id="meta-reference-faiss",
            pip_packages=EMBEDDING_DEPS + ["faiss-cpu"],
            module="llama_toolchain.memory.meta_reference.faiss",
            config_class="llama_toolchain.memory.meta_reference.faiss.FaissImplConfig",
        ),
        remote_provider_spec(
            api=Api.memory,
            adapter=AdapterSpec(
                adapter_id="chromadb",
                pip_packages=EMBEDDING_DEPS + ["chromadb-client"],
                module="llama_toolchain.memory.adapters.chroma",
            ),
        ),
        remote_provider_spec(
            api=Api.memory,
            adapter=AdapterSpec(
                adapter_id="pgvector",
                pip_packages=EMBEDDING_DEPS + ["psycopg2-binary"],
                module="llama_toolchain.memory.adapters.pgvector",
                config_class="llama_toolchain.memory.adapters.pgvector.PGVectorConfig",
            ),
        ),
    ]
