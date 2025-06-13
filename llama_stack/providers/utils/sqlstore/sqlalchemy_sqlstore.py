# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from collections.abc import Mapping
from typing import Any, Literal

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    inspect,
    select,
    text,
)
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from llama_stack.log import get_logger

from .api import ColumnDefinition, ColumnType, SqlStore
from .sqlstore import SqlAlchemySqlStoreConfig

logger = get_logger(name=__name__, category="sqlstore")

TYPE_MAPPING: dict[ColumnType, Any] = {
    ColumnType.INTEGER: Integer,
    ColumnType.STRING: String,
    ColumnType.FLOAT: Float,
    ColumnType.BOOLEAN: Boolean,
    ColumnType.DATETIME: DateTime,
    ColumnType.TEXT: Text,
    ColumnType.JSON: JSON,
}


class SqlAlchemySqlStoreImpl(SqlStore):
    def __init__(self, config: SqlAlchemySqlStoreConfig):
        self.config = config
        self.async_session = async_sessionmaker(create_async_engine(config.engine_str))
        self.metadata = MetaData()

    async def create_table(
        self,
        table: str,
        schema: Mapping[str, ColumnType | ColumnDefinition],
    ) -> None:
        if not schema:
            raise ValueError(f"No columns defined for table '{table}'.")

        sqlalchemy_columns: list[Column] = []

        for col_name, col_props in schema.items():
            col_type = None
            is_primary_key = False
            is_nullable = True

            if isinstance(col_props, ColumnType):
                col_type = col_props
            elif isinstance(col_props, ColumnDefinition):
                col_type = col_props.type
                is_primary_key = col_props.primary_key
                is_nullable = col_props.nullable

            sqlalchemy_type = TYPE_MAPPING.get(col_type)
            if not sqlalchemy_type:
                raise ValueError(f"Unsupported column type '{col_type}' for column '{col_name}'.")

            sqlalchemy_columns.append(
                Column(col_name, sqlalchemy_type, primary_key=is_primary_key, nullable=is_nullable)
            )

        if table not in self.metadata.tables:
            sqlalchemy_table = Table(table, self.metadata, *sqlalchemy_columns)
        else:
            sqlalchemy_table = self.metadata.tables[table]

        engine = create_async_engine(self.config.engine_str)
        async with engine.begin() as conn:
            await conn.run_sync(self.metadata.create_all, tables=[sqlalchemy_table], checkfirst=True)

    async def insert(self, table: str, data: Mapping[str, Any]) -> None:
        async with self.async_session() as session:
            await session.execute(self.metadata.tables[table].insert(), data)
            await session.commit()

    async def fetch_all(
        self,
        table: str,
        where: Mapping[str, Any] | None = None,
        where_sql: str | None = None,
        limit: int | None = None,
        order_by: list[tuple[str, Literal["asc", "desc"]]] | None = None,
    ) -> list[dict[str, Any]]:
        async with self.async_session() as session:
            query = select(self.metadata.tables[table])
            if where:
                for key, value in where.items():
                    query = query.where(self.metadata.tables[table].c[key] == value)
            if where_sql:
                query = query.where(text(where_sql))
            if limit:
                query = query.limit(limit)
            if order_by:
                if not isinstance(order_by, list):
                    raise ValueError(
                        f"order_by must be a list of tuples (column, order={['asc', 'desc']}), got {order_by}"
                    )
                for order in order_by:
                    if not isinstance(order, tuple):
                        raise ValueError(
                            f"order_by must be a list of tuples (column, order={['asc', 'desc']}), got {order_by}"
                        )
                    name, order_type = order
                    if order_type == "asc":
                        query = query.order_by(self.metadata.tables[table].c[name].asc())
                    elif order_type == "desc":
                        query = query.order_by(self.metadata.tables[table].c[name].desc())
                    else:
                        raise ValueError(f"Invalid order '{order_type}' for column '{name}'")
            result = await session.execute(query)
            return [dict(row._mapping) for row in result]

    async def fetch_one(
        self,
        table: str,
        where: Mapping[str, Any] | None = None,
        where_sql: str | None = None,
        order_by: list[tuple[str, Literal["asc", "desc"]]] | None = None,
    ) -> dict[str, Any] | None:
        rows = await self.fetch_all(table, where, where_sql, limit=1, order_by=order_by)
        if not rows:
            return None
        return rows[0]

    async def update(
        self,
        table: str,
        data: Mapping[str, Any],
        where: Mapping[str, Any],
    ) -> None:
        if not where:
            raise ValueError("where is required for update")

        async with self.async_session() as session:
            stmt = self.metadata.tables[table].update()
            for key, value in where.items():
                stmt = stmt.where(self.metadata.tables[table].c[key] == value)
            await session.execute(stmt, data)
            await session.commit()

    async def delete(self, table: str, where: Mapping[str, Any]) -> None:
        if not where:
            raise ValueError("where is required for delete")

        async with self.async_session() as session:
            stmt = self.metadata.tables[table].delete()
            for key, value in where.items():
                stmt = stmt.where(self.metadata.tables[table].c[key] == value)
            await session.execute(stmt)
            await session.commit()

    async def add_column_if_not_exists(
        self,
        table: str,
        column_name: str,
        column_type: ColumnType,
        nullable: bool = True,
    ) -> None:
        """Add a column to an existing table if the column doesn't already exist."""
        engine = create_async_engine(self.config.engine_str)

        try:
            inspector = inspect(engine)

            table_names = inspector.get_table_names()
            if table not in table_names:
                return

            existing_columns = inspector.get_columns(table)
            column_names = [col["name"] for col in existing_columns]

            if column_name in column_names:
                return

            sqlalchemy_type = TYPE_MAPPING.get(column_type)
            if not sqlalchemy_type:
                raise ValueError(f"Unsupported column type '{column_type}' for column '{column_name}'.")

            # Create the ALTER TABLE statement
            # Note: We need to get the dialect-specific type name
            dialect = engine.dialect
            type_impl = sqlalchemy_type()
            compiled_type = type_impl.compile(dialect=dialect)

            nullable_clause = "" if nullable else " NOT NULL"
            add_column_sql = text(f"ALTER TABLE {table} ADD COLUMN {column_name} {compiled_type}{nullable_clause}")

            async with engine.begin() as conn:
                await conn.execute(add_column_sql)

        except Exception:
            # If any error occurs during migration, log it but don't fail
            # The table creation will handle adding the column
            pass
