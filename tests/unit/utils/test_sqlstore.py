# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from tempfile import TemporaryDirectory

import pytest

from llama_stack.providers.utils.sqlstore.api import ColumnType
from llama_stack.providers.utils.sqlstore.sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl
from llama_stack.providers.utils.sqlstore.sqlstore import SqliteSqlStoreConfig


@pytest.mark.asyncio
async def test_sqlite_sqlstore():
    with TemporaryDirectory() as tmp_dir:
        db_name = "test.db"
        sqlstore = SqlAlchemySqlStoreImpl(
            SqliteSqlStoreConfig(
                db_path=tmp_dir + "/" + db_name,
            )
        )
        await sqlstore.create_table(
            table="test",
            schema={
                "id": ColumnType.INTEGER,
                "name": ColumnType.STRING,
            },
        )
        await sqlstore.insert("test", {"id": 1, "name": "test"})
        await sqlstore.insert("test", {"id": 12, "name": "test12"})
        rows = await sqlstore.fetch_all("test")
        assert rows == [{"id": 1, "name": "test"}, {"id": 12, "name": "test12"}]

        row = await sqlstore.fetch_one("test", {"id": 1})
        assert row == {"id": 1, "name": "test"}

        row = await sqlstore.fetch_one("test", {"name": "test12"})
        assert row == {"id": 12, "name": "test12"}

        # order by
        rows = await sqlstore.fetch_all("test", order_by=[("id", "asc")])
        assert rows == [{"id": 1, "name": "test"}, {"id": 12, "name": "test12"}]

        rows = await sqlstore.fetch_all("test", order_by=[("id", "desc")])
        assert rows == [{"id": 12, "name": "test12"}, {"id": 1, "name": "test"}]

        # limit
        rows = await sqlstore.fetch_all("test", limit=1)
        assert rows == [{"id": 1, "name": "test"}]

        # update
        await sqlstore.update("test", {"name": "test123"}, {"id": 1})
        row = await sqlstore.fetch_one("test", {"id": 1})
        assert row == {"id": 1, "name": "test123"}

        # delete
        await sqlstore.delete("test", {"id": 1})
        rows = await sqlstore.fetch_all("test")
        assert rows == [{"id": 12, "name": "test12"}]
