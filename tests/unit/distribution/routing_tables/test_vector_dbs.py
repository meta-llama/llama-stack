# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Unit tests for the routing tables vector_dbs

import time
from unittest.mock import AsyncMock

import pytest

from llama_stack.apis.datatypes import Api
from llama_stack.apis.models import ModelType
from llama_stack.apis.vector_dbs.vector_dbs import VectorDB
from llama_stack.apis.vector_io.vector_io import (
    VectorStoreContent,
    VectorStoreDeleteResponse,
    VectorStoreFileContentsResponse,
    VectorStoreFileCounts,
    VectorStoreFileDeleteResponse,
    VectorStoreFileObject,
    VectorStoreObject,
    VectorStoreSearchResponsePage,
)
from llama_stack.core.access_control.datatypes import AccessRule, Scope
from llama_stack.core.datatypes import User
from llama_stack.core.request_headers import request_provider_data_context
from llama_stack.core.routing_tables.vector_dbs import VectorDBsRoutingTable
from tests.unit.distribution.routers.test_routing_tables import Impl, InferenceImpl, ModelsRoutingTable


class VectorDBImpl(Impl):
    def __init__(self):
        super().__init__(Api.vector_io)

    async def register_vector_db(self, vector_db: VectorDB):
        return vector_db

    async def unregister_vector_db(self, vector_db_id: str):
        return vector_db_id

    async def openai_retrieve_vector_store(self, vector_store_id):
        return VectorStoreObject(
            id=vector_store_id,
            name="Test Store",
            created_at=int(time.time()),
            file_counts=VectorStoreFileCounts(completed=0, cancelled=0, failed=0, in_progress=0, total=0),
        )

    async def openai_update_vector_store(self, vector_store_id, **kwargs):
        return VectorStoreObject(
            id=vector_store_id,
            name="Updated Store",
            created_at=int(time.time()),
            file_counts=VectorStoreFileCounts(completed=0, cancelled=0, failed=0, in_progress=0, total=0),
        )

    async def openai_delete_vector_store(self, vector_store_id):
        return VectorStoreDeleteResponse(id=vector_store_id, object="vector_store.deleted", deleted=True)

    async def openai_search_vector_store(self, vector_store_id, query, **kwargs):
        return VectorStoreSearchResponsePage(
            object="vector_store.search_results.page", search_query="query", data=[], has_more=False, next_page=None
        )

    async def openai_attach_file_to_vector_store(self, vector_store_id, file_id, **kwargs):
        return VectorStoreFileObject(
            id=file_id,
            status="completed",
            chunking_strategy={"type": "auto"},
            created_at=int(time.time()),
            vector_store_id=vector_store_id,
        )

    async def openai_list_files_in_vector_store(self, vector_store_id, **kwargs):
        return [
            VectorStoreFileObject(
                id="1",
                status="completed",
                chunking_strategy={"type": "auto"},
                created_at=int(time.time()),
                vector_store_id=vector_store_id,
            )
        ]

    async def openai_retrieve_vector_store_file(self, vector_store_id, file_id):
        return VectorStoreFileObject(
            id=file_id,
            status="completed",
            chunking_strategy={"type": "auto"},
            created_at=int(time.time()),
            vector_store_id=vector_store_id,
        )

    async def openai_retrieve_vector_store_file_contents(self, vector_store_id, file_id):
        return VectorStoreFileContentsResponse(
            file_id=file_id,
            filename="Sample File name",
            attributes={"key": "value"},
            content=[VectorStoreContent(type="text", text="Sample content")],
        )

    async def openai_update_vector_store_file(self, vector_store_id, file_id, **kwargs):
        return VectorStoreFileObject(
            id=file_id,
            status="completed",
            chunking_strategy={"type": "auto"},
            created_at=int(time.time()),
            vector_store_id=vector_store_id,
        )

    async def openai_delete_vector_store_file(self, vector_store_id, file_id):
        return VectorStoreFileDeleteResponse(id=file_id, deleted=True)


async def test_vectordbs_routing_table(cached_disk_dist_registry):
    table = VectorDBsRoutingTable({"test_provider": VectorDBImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    m_table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, {})
    await m_table.initialize()
    await m_table.register_model(
        model_id="test-model",
        provider_id="test_provider",
        metadata={"embedding_dimension": 128},
        model_type=ModelType.embedding,
    )

    # Register multiple vector databases and verify listing
    await table.register_vector_db(vector_db_id="test-vectordb", embedding_model="test-model")
    await table.register_vector_db(vector_db_id="test-vectordb-2", embedding_model="test-model")
    vector_dbs = await table.list_vector_dbs()

    assert len(vector_dbs.data) == 2
    vector_db_ids = {v.identifier for v in vector_dbs.data}
    assert "test-vectordb" in vector_db_ids
    assert "test-vectordb-2" in vector_db_ids

    await table.unregister_vector_db(vector_db_id="test-vectordb")
    await table.unregister_vector_db(vector_db_id="test-vectordb-2")

    vector_dbs = await table.list_vector_dbs()
    assert len(vector_dbs.data) == 0


async def test_openai_vector_stores_routing_table_roles(cached_disk_dist_registry):
    impl = VectorDBImpl()
    impl.openai_retrieve_vector_store = AsyncMock(return_value="OK")
    table = VectorDBsRoutingTable({"test_provider": impl}, cached_disk_dist_registry, policy=[])
    m_table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, policy=[])
    authorized_table = "vs1"
    authorized_team = "team1"
    unauthorized_team = "team2"

    await m_table.initialize()
    await m_table.register_model(
        model_id="test-model",
        provider_id="test_provider",
        metadata={"embedding_dimension": 128},
        model_type=ModelType.embedding,
    )

    authorized_user = User(principal="alice", attributes={"roles": [authorized_team]})
    with request_provider_data_context({}, authorized_user):
        _ = await table.register_vector_db(vector_db_id="vs1", embedding_model="test-model")

    # Authorized reader
    with request_provider_data_context({}, authorized_user):
        res = await table.openai_retrieve_vector_store(authorized_table)
    assert res == "OK"

    # Authorized updater
    impl.openai_update_vector_store_file = AsyncMock(return_value="UPDATED")
    with request_provider_data_context({}, authorized_user):
        res = await table.openai_update_vector_store_file(authorized_table, file_id="file1", attributes={"foo": "bar"})
    assert res == "UPDATED"

    # Unauthorized reader
    unauthorized_user = User(principal="eve", attributes={"roles": [unauthorized_team]})
    with request_provider_data_context({}, unauthorized_user):
        with pytest.raises(ValueError):
            await table.openai_retrieve_vector_store(authorized_table)

    # Unauthorized updater
    with request_provider_data_context({}, unauthorized_user):
        with pytest.raises(ValueError):
            await table.openai_update_vector_store_file(authorized_table, file_id="file1", attributes={"foo": "bar"})

    # Authorized deleter
    impl.openai_delete_vector_store_file = AsyncMock(return_value="DELETED")
    with request_provider_data_context({}, authorized_user):
        res = await table.openai_delete_vector_store_file(authorized_table, file_id="file1")
    assert res == "DELETED"

    # Unauthorized deleter
    with request_provider_data_context({}, unauthorized_user):
        with pytest.raises(ValueError):
            await table.openai_delete_vector_store_file(authorized_table, file_id="file1")


async def test_openai_vector_stores_routing_table_actions(cached_disk_dist_registry):
    impl = VectorDBImpl()

    policy = [
        AccessRule(permit=Scope(actions=["create", "read", "update", "delete"]), when="user with admin in roles"),
        AccessRule(permit=Scope(actions=["read"]), when="user with reader in roles"),
    ]

    table = VectorDBsRoutingTable({"test_provider": impl}, cached_disk_dist_registry, policy=policy)
    m_table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, policy=[])

    vector_db_id = "vs1"
    file_id = "file-1"

    admin_user = User(principal="admin", attributes={"roles": ["admin"]})
    read_only_user = User(principal="reader", attributes={"roles": ["reader"]})
    no_access_user = User(principal="outsider", attributes={"roles": ["no_access"]})

    await m_table.initialize()
    await m_table.register_model(
        model_id="test-model",
        provider_id="test_provider",
        metadata={"embedding_dimension": 128},
        model_type=ModelType.embedding,
    )

    with request_provider_data_context({}, admin_user):
        await table.register_vector_db(vector_db_id=vector_db_id, embedding_model="test-model")

    read_methods = [
        (table.openai_retrieve_vector_store, (vector_db_id,), {}),
        (table.openai_search_vector_store, (vector_db_id, "query"), {}),
        (table.openai_list_files_in_vector_store, (vector_db_id,), {}),
        (table.openai_retrieve_vector_store_file, (vector_db_id, file_id), {}),
        (table.openai_retrieve_vector_store_file_contents, (vector_db_id, file_id), {}),
    ]
    update_methods = [
        (table.openai_update_vector_store, (vector_db_id,), {"name": "Updated DB"}),
        (table.openai_attach_file_to_vector_store, (vector_db_id, file_id), {}),
        (table.openai_update_vector_store_file, (vector_db_id, file_id), {"attributes": {"key": "value"}}),
    ]
    delete_methods = [
        (table.openai_delete_vector_store_file, (vector_db_id, file_id), {}),
        (table.openai_delete_vector_store, (vector_db_id,), {}),
    ]

    for user in [admin_user, read_only_user]:
        with request_provider_data_context({}, user):
            for method, args, kwargs in read_methods:
                result = await method(*args, **kwargs)
                assert result is not None, f"Read operation failed with user {user.principal}"

    with request_provider_data_context({}, no_access_user):
        for method, args, kwargs in read_methods:
            with pytest.raises(ValueError):
                await method(*args, **kwargs)

    with request_provider_data_context({}, admin_user):
        for method, args, kwargs in update_methods:
            result = await method(*args, **kwargs)
            assert result is not None, "Update operation failed with admin user"

    with request_provider_data_context({}, admin_user):
        for method, args, kwargs in delete_methods:
            result = await method(*args, **kwargs)
            assert result is not None, "Delete operation failed with admin user"

    for user in [read_only_user, no_access_user]:
        with request_provider_data_context({}, user):
            for method, args, kwargs in delete_methods:
                with pytest.raises(ValueError):
                    await method(*args, **kwargs)
