# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import create_autospec, patch

import pytest

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.memory.memory import *  # noqa: F403
from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.providers.datatypes import MemoryBanksProtocolPrivate


class MemoryImplFake(Memory, MemoryBanksProtocolPrivate): ...


class MemoryAdapterFake(
    Memory, NeedsRequestProviderData, MemoryBanksProtocolPrivate
): ...


class MethodStubs:
    QUERY_DOCUMENTS_RETURN_VALUES = [
        QueryDocumentsResponse(
            chunks=[Chunk(content="Python", token_count=1, document_id="")],
            scores=[0.1],
        ),
        QueryDocumentsResponse(
            chunks=[Chunk(content="neural networks", token_count=2, document_id="")],
            scores=[0.1],
        ),
        QueryDocumentsResponse(
            chunks=[
                Chunk(content="chunk-1", token_count=1, document_id=""),
                Chunk(content="chunk-2", token_count=1, document_id=""),
            ],
            scores=[0.1, 0.5],
        ),
        QueryDocumentsResponse(
            chunks=[Chunk(content="Python", token_count=1, document_id="")],
            scores=[0.5],
        ),
    ]


@pytest.fixture(scope="session")
def memory_faiss_mocks(request):
    with patch(
        "llama_stack.providers.inline.memory.faiss.get_provider_impl",
        autospec=True,
    ) as get_adapter_impl_mock:  # noqa N806
        impl_mock = create_autospec(MemoryImplFake)
        impl_mock.query_documents.side_effect = (
            MethodStubs.QUERY_DOCUMENTS_RETURN_VALUES
        )
        get_adapter_impl_mock.return_value = impl_mock
        yield


@pytest.fixture(scope="session")
def memory_pgvector_mocks(request):
    with patch(
        "llama_stack.providers.remote.memory.pgvector.get_adapter_impl",
        autospec=True,
    ) as get_adapter_impl_mock:  # noqa N806
        adapter_mock = create_autospec(MemoryAdapterFake)
        adapter_mock.query_documents.side_effect = (
            MethodStubs.QUERY_DOCUMENTS_RETURN_VALUES
        )
        get_adapter_impl_mock.return_value = adapter_mock
        yield


@pytest.fixture(scope="session")
def memory_weaviate_mocks(request):
    with patch(
        "llama_stack.providers.remote.memory.weaviate.get_adapter_impl",
        autospec=True,
    ) as get_adapter_impl_mock:  # noqa N806
        adapter_mock = create_autospec(MemoryAdapterFake)
        adapter_mock.query_documents.side_effect = (
            MethodStubs.QUERY_DOCUMENTS_RETURN_VALUES
        )
        get_adapter_impl_mock.return_value = adapter_mock
        yield


@pytest.fixture(scope="session")
def memory_chroma_mocks(request):
    with patch(
        "llama_stack.providers.remote.memory.chroma.get_adapter_impl",
        autospec=True,
    ) as get_adapter_impl_mock:  # noqa N806
        adapter_mock = create_autospec(MemoryAdapterFake)
        adapter_mock.query_documents.side_effect = (
            MethodStubs.QUERY_DOCUMENTS_RETURN_VALUES
        )
        get_adapter_impl_mock.return_value = adapter_mock
        yield
