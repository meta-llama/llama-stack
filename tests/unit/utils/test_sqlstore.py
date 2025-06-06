# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from tempfile import TemporaryDirectory

import pytest
import sqlalchemy

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


@pytest.mark.asyncio
async def test_where_sql_basic_functionality():
    """Test where_sql parameter with basic SQL queries"""
    with TemporaryDirectory() as tmp_dir:
        db_name = "test_where_sql.db"
        sqlstore = SqlAlchemySqlStoreImpl(
            SqliteSqlStoreConfig(
                db_path=tmp_dir + "/" + db_name,
            )
        )

        # Create test table with more diverse data
        await sqlstore.create_table(
            table="users",
            schema={
                "id": ColumnType.INTEGER,
                "name": ColumnType.STRING,
                "age": ColumnType.INTEGER,
                "city": ColumnType.STRING,
            },
        )

        # Insert test data
        test_users = [
            {"id": 1, "name": "Alice", "age": 25, "city": "New York"},
            {"id": 2, "name": "Bob", "age": 30, "city": "San Francisco"},
            {"id": 3, "name": "Charlie", "age": 35, "city": "New York"},
            {"id": 4, "name": "Diana", "age": 28, "city": "Boston"},
        ]

        for user in test_users:
            await sqlstore.insert("users", user)

        # Test simple where_sql queries
        rows = await sqlstore.fetch_all("users", where_sql="age > 28")
        assert len(rows) == 2
        assert {row["name"] for row in rows} == {"Bob", "Charlie"}

        # Test where_sql with LIKE
        rows = await sqlstore.fetch_all("users", where_sql="city LIKE '%New%'")
        assert len(rows) == 2
        assert {row["name"] for row in rows} == {"Alice", "Charlie"}

        # Test where_sql with AND conditions
        rows = await sqlstore.fetch_all("users", where_sql="age >= 30 AND city = 'New York'")
        assert len(rows) == 1
        assert rows[0]["name"] == "Charlie"

        # Test where_sql with OR conditions
        rows = await sqlstore.fetch_all("users", where_sql="age < 27 OR city = 'Boston'")
        assert len(rows) == 2
        assert {row["name"] for row in rows} == {"Alice", "Diana"}


@pytest.mark.asyncio
async def test_where_sql_combined_with_structured_where():
    """Test combination of structured where and where_sql parameters"""
    with TemporaryDirectory() as tmp_dir:
        db_name = "test_combined.db"
        sqlstore = SqlAlchemySqlStoreImpl(
            SqliteSqlStoreConfig(
                db_path=tmp_dir + "/" + db_name,
            )
        )

        await sqlstore.create_table(
            table="products",
            schema={
                "id": ColumnType.INTEGER,
                "category": ColumnType.STRING,
                "price": ColumnType.FLOAT,
                "in_stock": ColumnType.BOOLEAN,
            },
        )

        # Insert test data
        products = [
            {"id": 1, "category": "electronics", "price": 99.99, "in_stock": True},
            {"id": 2, "category": "electronics", "price": 199.99, "in_stock": False},
            {"id": 3, "category": "books", "price": 15.99, "in_stock": True},
            {"id": 4, "category": "electronics", "price": 299.99, "in_stock": True},
        ]

        for product in products:
            await sqlstore.insert("products", product)

        # Test combining structured where (category) with where_sql (price range)
        rows = await sqlstore.fetch_all(
            "products", where={"category": "electronics"}, where_sql="price > 150 AND in_stock = 1"
        )
        assert len(rows) == 1
        assert rows[0]["id"] == 4
        assert rows[0]["price"] == 299.99


@pytest.mark.asyncio
async def test_where_sql_with_fetch_one():
    """Test where_sql parameter with fetch_one method"""
    with TemporaryDirectory() as tmp_dir:
        db_name = "test_fetch_one.db"
        sqlstore = SqlAlchemySqlStoreImpl(
            SqliteSqlStoreConfig(
                db_path=tmp_dir + "/" + db_name,
            )
        )

        await sqlstore.create_table(
            table="employees",
            schema={
                "id": ColumnType.INTEGER,
                "name": ColumnType.STRING,
                "department": ColumnType.STRING,
                "salary": ColumnType.INTEGER,
            },
        )

        employees = [
            {"id": 1, "name": "John", "department": "Engineering", "salary": 90000},
            {"id": 2, "name": "Jane", "department": "Marketing", "salary": 75000},
            {"id": 3, "name": "Mike", "department": "Engineering", "salary": 95000},
        ]

        for emp in employees:
            await sqlstore.insert("employees", emp)

        # Test fetch_one with where_sql
        row = await sqlstore.fetch_one("employees", where_sql="salary > 90000")
        assert row is not None
        assert row["name"] in ["John", "Mike"]  # Could be either depending on order

        # Test fetch_one with where_sql that returns no results
        row = await sqlstore.fetch_one("employees", where_sql="salary > 100000")
        assert row is None

        # Test fetch_one with combined where and where_sql
        row = await sqlstore.fetch_one("employees", where={"department": "Engineering"}, where_sql="salary >= 95000")
        assert row is not None
        assert row["name"] == "Mike"


@pytest.mark.asyncio
async def test_where_sql_edge_cases():
    """Test edge cases for where_sql parameter"""
    with TemporaryDirectory() as tmp_dir:
        db_name = "test_edge_cases.db"
        sqlstore = SqlAlchemySqlStoreImpl(
            SqliteSqlStoreConfig(
                db_path=tmp_dir + "/" + db_name,
            )
        )

        await sqlstore.create_table(
            table="test_table",
            schema={
                "id": ColumnType.INTEGER,
                "value": ColumnType.STRING,
            },
        )

        await sqlstore.insert("test_table", {"id": 1, "value": "test"})

        # Test with None where_sql (should work like before)
        rows = await sqlstore.fetch_all("test_table", where_sql=None)
        assert len(rows) == 1

        # Test with empty string where_sql (should return all rows)
        rows = await sqlstore.fetch_all("test_table", where_sql="")
        assert len(rows) == 1

        # Test fetch_one with None where_sql
        row = await sqlstore.fetch_one("test_table", where_sql=None)
        assert row is not None
        assert row["id"] == 1


@pytest.mark.asyncio
async def test_where_sql_ordering_and_limits():
    """Test where_sql with ordering and limit parameters"""
    with TemporaryDirectory() as tmp_dir:
        db_name = "test_ordering.db"
        sqlstore = SqlAlchemySqlStoreImpl(
            SqliteSqlStoreConfig(
                db_path=tmp_dir + "/" + db_name,
            )
        )

        await sqlstore.create_table(
            table="scores",
            schema={
                "id": ColumnType.INTEGER,
                "player": ColumnType.STRING,
                "score": ColumnType.INTEGER,
                "game": ColumnType.STRING,
            },
        )

        scores = [
            {"id": 1, "player": "Alice", "score": 100, "game": "puzzle"},
            {"id": 2, "player": "Bob", "score": 85, "game": "puzzle"},
            {"id": 3, "player": "Charlie", "score": 95, "game": "puzzle"},
            {"id": 4, "player": "Alice", "score": 120, "game": "action"},
            {"id": 5, "player": "Bob", "score": 75, "game": "action"},
        ]

        for score in scores:
            await sqlstore.insert("scores", score)

        # Test where_sql with ordering
        rows = await sqlstore.fetch_all("scores", where_sql="score >= 90", order_by=[("score", "desc")])
        assert len(rows) == 3
        assert rows[0]["score"] == 120  # Alice action
        assert rows[1]["score"] == 100  # Alice puzzle
        assert rows[2]["score"] == 95  # Charlie puzzle

        # Test where_sql with limit
        rows = await sqlstore.fetch_all("scores", where_sql="game = 'puzzle'", order_by=[("score", "desc")], limit=2)
        assert len(rows) == 2
        assert rows[0]["player"] == "Alice"
        assert rows[1]["player"] == "Charlie"


@pytest.mark.asyncio
async def test_where_sql_error_handling():
    """Test error handling for malformed SQL in where_sql"""
    with TemporaryDirectory() as tmp_dir:
        db_name = "test_errors.db"
        sqlstore = SqlAlchemySqlStoreImpl(
            SqliteSqlStoreConfig(
                db_path=tmp_dir + "/" + db_name,
            )
        )

        await sqlstore.create_table(
            table="test_table",
            schema={
                "id": ColumnType.INTEGER,
                "name": ColumnType.STRING,
            },
        )

        await sqlstore.insert("test_table", {"id": 1, "name": "test"})

        # Test malformed SQL - should raise an exception
        with pytest.raises(sqlalchemy.exc.OperationalError):
            await sqlstore.fetch_all("test_table", where_sql="INVALID SQL SYNTAX")

        # Test nonexistent column - should raise an exception
        with pytest.raises(sqlalchemy.exc.OperationalError):
            await sqlstore.fetch_all("test_table", where_sql="nonexistent_column = 'value'")

        # Test that well-formed SQL still works after errors
        rows = await sqlstore.fetch_all("test_table", where_sql="id = 1")
        assert len(rows) == 1
        assert rows[0]["name"] == "test"
