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
    "tqdm",
    "numpy",
    "scikit-learn",
    "scipy",
    "nltk",
    "sentencepiece",
    "transformers",
    # this happens to work because special dependencies are always installed last
    # so if there was a regular torch installed first, this would be ignored
    # we need a better way to do this to identify potential conflicts, etc.
    # for now, this lets us significantly reduce the size of the container which
    # does not have any "local" inference code (and hence does not need GPU-enabled torch)
    "torch --index-url https://download.pytorch.org/whl/cpu",
    "sentence-transformers --no-deps",
]


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.memory,
            provider_type="meta-reference",
            pip_packages=EMBEDDING_DEPS + ["faiss-cpu"],
            module="llama_stack.providers.impls.meta_reference.memory",
            config_class="llama_stack.providers.impls.meta_reference.memory.FaissImplConfig",
        ),
        remote_provider_spec(
            Api.memory,
            AdapterSpec(
                adapter_type="chromadb",
                pip_packages=EMBEDDING_DEPS + ["chromadb-client"],
                module="llama_stack.providers.adapters.memory.chroma",
            ),
        ),
        remote_provider_spec(
            Api.memory,
            AdapterSpec(
                adapter_type="pgvector",
                pip_packages=EMBEDDING_DEPS + ["psycopg2-binary"],
                module="llama_stack.providers.adapters.memory.pgvector",
                config_class="llama_stack.providers.adapters.memory.pgvector.PGVectorConfig",
            ),
        ),
        remote_provider_spec(
            Api.memory,
            AdapterSpec(
                adapter_type="weaviate",
                pip_packages=EMBEDDING_DEPS + ["weaviate-client"],
                module="llama_stack.providers.adapters.memory.weaviate",
                config_class="llama_stack.providers.adapters.memory.weaviate.WeaviateConfig",
                provider_data_validator="llama_stack.providers.adapters.memory.weaviate.WeaviateRequestProviderData",
            ),
        ),
        remote_provider_spec(
            api=Api.memory,
            adapter=AdapterSpec(
                adapter_type="sample",
                pip_packages=[],
                module="llama_stack.providers.adapters.memory.sample",
                config_class="llama_stack.providers.adapters.memory.sample.SampleConfig",
            ),
        ),
        remote_provider_spec(
            Api.memory,
            AdapterSpec(
                adapter_type="qdrant",
                pip_packages=EMBEDDING_DEPS + ["qdrant-client"],
                module="llama_stack.providers.adapters.memory.qdrant",
                config_class="llama_stack.providers.adapters.memory.qdrant.QdrantConfig",
            ),
        ),
    ]
