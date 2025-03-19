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


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.vector_io,
            provider_type="inline::meta-reference",
            pip_packages=["faiss-cpu"],
            module="llama_stack.providers.inline.vector_io.faiss",
            config_class="llama_stack.providers.inline.vector_io.faiss.FaissVectorIOConfig",
            deprecation_warning="Please use the `inline::faiss` provider instead.",
            api_dependencies=[Api.inference],
        ),
        InlineProviderSpec(
            api=Api.vector_io,
            provider_type="inline::faiss",
            pip_packages=["faiss-cpu"],
            module="llama_stack.providers.inline.vector_io.faiss",
            config_class="llama_stack.providers.inline.vector_io.faiss.FaissVectorIOConfig",
            api_dependencies=[Api.inference],
        ),
        # NOTE: sqlite-vec cannot be bundled into the container image because it does not have a
        # source distribution and the wheels are not available for all platforms.
        InlineProviderSpec(
            api=Api.vector_io,
            provider_type="inline::sqlite-vec",
            pip_packages=["sqlite-vec"],
            module="llama_stack.providers.inline.vector_io.sqlite_vec",
            config_class="llama_stack.providers.inline.vector_io.sqlite_vec.SQLiteVectorIOConfig",
            api_dependencies=[Api.inference],
        ),
        InlineProviderSpec(
            api=Api.vector_io,
            provider_type="inline::sqlite_vec",
            pip_packages=["sqlite-vec"],
            module="llama_stack.providers.inline.vector_io.sqlite_vec",
            config_class="llama_stack.providers.inline.vector_io.sqlite_vec.SQLiteVectorIOConfig",
            deprecation_warning="Please use the `inline::sqlite-vec` provider (notice the hyphen instead of underscore) instead.",
            api_dependencies=[Api.inference],
        ),
        remote_provider_spec(
            Api.vector_io,
            AdapterSpec(
                adapter_type="chromadb",
                pip_packages=["chromadb-client"],
                module="llama_stack.providers.remote.vector_io.chroma",
                config_class="llama_stack.providers.remote.vector_io.chroma.ChromaVectorIOConfig",
            ),
            api_dependencies=[Api.inference],
        ),
        InlineProviderSpec(
            api=Api.vector_io,
            provider_type="inline::chromadb",
            pip_packages=["chromadb"],
            module="llama_stack.providers.inline.vector_io.chroma",
            config_class="llama_stack.providers.inline.vector_io.chroma.ChromaVectorIOConfig",
            api_dependencies=[Api.inference],
        ),
        remote_provider_spec(
            Api.vector_io,
            AdapterSpec(
                adapter_type="pgvector",
                pip_packages=["psycopg2-binary"],
                module="llama_stack.providers.remote.vector_io.pgvector",
                config_class="llama_stack.providers.remote.vector_io.pgvector.PGVectorVectorIOConfig",
            ),
            api_dependencies=[Api.inference],
        ),
        remote_provider_spec(
            Api.vector_io,
            AdapterSpec(
                adapter_type="weaviate",
                pip_packages=["weaviate-client"],
                module="llama_stack.providers.remote.vector_io.weaviate",
                config_class="llama_stack.providers.remote.vector_io.weaviate.WeaviateVectorIOConfig",
                provider_data_validator="llama_stack.providers.remote.vector_io.weaviate.WeaviateRequestProviderData",
            ),
            api_dependencies=[Api.inference],
        ),
        InlineProviderSpec(
            api=Api.vector_io,
            provider_type="inline::qdrant",
            pip_packages=["qdrant-client"],
            module="llama_stack.providers.inline.vector_io.qdrant",
            config_class="llama_stack.providers.inline.vector_io.qdrant.QdrantVectorIOConfig",
            api_dependencies=[Api.inference],
        ),
        remote_provider_spec(
            Api.vector_io,
            AdapterSpec(
                adapter_type="qdrant",
                pip_packages=["qdrant-client"],
                module="llama_stack.providers.remote.vector_io.qdrant",
                config_class="llama_stack.providers.remote.vector_io.qdrant.QdrantVectorIOConfig",
            ),
            api_dependencies=[Api.inference],
        ),
        remote_provider_spec(
            Api.vector_io,
            AdapterSpec(
                adapter_type="milvus",
                pip_packages=["pymilvus"],
                module="llama_stack.providers.remote.vector_io.milvus",
                config_class="llama_stack.providers.remote.vector_io.milvus.MilvusVectorIOConfig",
            ),
            api_dependencies=[Api.inference],
        ),
        InlineProviderSpec(
            api=Api.vector_io,
            provider_type="inline::milvus",
            pip_packages=["pymilvus"],
            module="llama_stack.providers.inline.vector_io.milvus",
            config_class="llama_stack.providers.inline.vector_io.milvus.MilvusVectorIOConfig",
            api_dependencies=[Api.inference],
        ),
    ]
