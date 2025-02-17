# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.providers.datatypes import (
    AdapterSpec,
    Api,
    InlineProviderSpec,
    ProviderSpec,
    remote_provider_spec,
)

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
    "torch torchvision --index-url https://download.pytorch.org/whl/cpu",
    "sentence-transformers --no-deps",
]


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.vector_io,
            provider_type="inline::meta-reference",
            pip_packages=EMBEDDING_DEPS + ["faiss-cpu"],
            module="llama_stack.providers.inline.vector_io.faiss",
            config_class="llama_stack.providers.inline.vector_io.faiss.FaissVectorIOConfig",
            deprecation_warning="Please use the `inline::faiss` provider instead.",
            api_dependencies=[Api.inference],
        ),
        InlineProviderSpec(
            api=Api.vector_io,
            provider_type="inline::faiss",
            pip_packages=EMBEDDING_DEPS + ["faiss-cpu"],
            module="llama_stack.providers.inline.vector_io.faiss",
            config_class="llama_stack.providers.inline.vector_io.faiss.FaissVectorIOConfig",
            api_dependencies=[Api.inference],
        ),
        InlineProviderSpec(
            api=Api.vector_io,
            provider_type="inline::sqlite_vec",
            pip_packages=EMBEDDING_DEPS + ["sqlite-vec"],
            module="llama_stack.providers.inline.vector_io.sqlite_vec",
            config_class="llama_stack.providers.inline.vector_io.sqlite_vec.SQLiteVectorIOConfig",
            api_dependencies=[Api.inference],
        ),
        remote_provider_spec(
            Api.vector_io,
            AdapterSpec(
                adapter_type="chromadb",
                pip_packages=EMBEDDING_DEPS + ["chromadb-client"],
                module="llama_stack.providers.remote.vector_io.chroma",
                config_class="llama_stack.providers.remote.vector_io.chroma.ChromaVectorIOConfig",
            ),
            api_dependencies=[Api.inference],
        ),
        InlineProviderSpec(
            api=Api.vector_io,
            provider_type="inline::chromadb",
            pip_packages=EMBEDDING_DEPS + ["chromadb"],
            module="llama_stack.providers.inline.vector_io.chroma",
            config_class="llama_stack.providers.inline.vector_io.chroma.ChromaVectorIOConfig",
            api_dependencies=[Api.inference],
        ),
        remote_provider_spec(
            Api.vector_io,
            AdapterSpec(
                adapter_type="pgvector",
                pip_packages=EMBEDDING_DEPS + ["psycopg2-binary"],
                module="llama_stack.providers.remote.vector_io.pgvector",
                config_class="llama_stack.providers.remote.vector_io.pgvector.PGVectorVectorIOConfig",
            ),
            api_dependencies=[Api.inference],
        ),
        remote_provider_spec(
            Api.vector_io,
            AdapterSpec(
                adapter_type="weaviate",
                pip_packages=EMBEDDING_DEPS + ["weaviate-client"],
                module="llama_stack.providers.remote.vector_io.weaviate",
                config_class="llama_stack.providers.remote.vector_io.weaviate.WeaviateVectorIOConfig",
                provider_data_validator="llama_stack.providers.remote.vector_io.weaviate.WeaviateRequestProviderData",
            ),
            api_dependencies=[Api.inference],
        ),
        remote_provider_spec(
            api=Api.vector_io,
            adapter=AdapterSpec(
                adapter_type="sample",
                pip_packages=[],
                module="llama_stack.providers.remote.vector_io.sample",
                config_class="llama_stack.providers.remote.vector_io.sample.SampleVectorIOConfig",
            ),
            api_dependencies=[],
        ),
        remote_provider_spec(
            Api.vector_io,
            AdapterSpec(
                adapter_type="qdrant",
                pip_packages=EMBEDDING_DEPS + ["qdrant-client"],
                module="llama_stack.providers.remote.vector_io.qdrant",
                config_class="llama_stack.providers.remote.vector_io.qdrant.QdrantVectorIOConfig",
            ),
            api_dependencies=[Api.inference],
        ),
    ]
