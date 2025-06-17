# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Mapping
from enum import Enum
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from llama_stack.apis.common.responses import PaginatedResponse


class ColumnType(Enum):
    INTEGER = "INTEGER"
    STRING = "STRING"
    TEXT = "TEXT"
    FLOAT = "FLOAT"
    BOOLEAN = "BOOLEAN"
    JSON = "JSON"
    DATETIME = "DATETIME"


class ColumnDefinition(BaseModel):
    type: ColumnType
    primary_key: bool = False
    nullable: bool = True
    default: Any = None


class SqlStore(Protocol):
    """
    A protocol for a SQL store.
    """

    async def create_table(self, table: str, schema: Mapping[str, ColumnType | ColumnDefinition]) -> None:
        """
        Create a table.
        """
        pass

    async def insert(self, table: str, data: Mapping[str, Any]) -> None:
        """
        Insert a row into a table.
        """
        pass

    async def fetch_all(
        self,
        table: str,
        where: Mapping[str, Any] | None = None,
        limit: int | None = None,
        order_by: list[tuple[str, Literal["asc", "desc"]]] | None = None,
        cursor: tuple[str, str] | None = None,
    ) -> PaginatedResponse:
        """
        Fetch all rows from a table with optional cursor-based pagination.

        :param table: The table name
        :param where: WHERE conditions
        :param limit: Maximum number of records to return
        :param order_by: List of (column, order) tuples for sorting
        :param cursor: Tuple of (key_column, cursor_id) for pagination (None for first page)
                      Requires order_by with exactly one column when used
        :return: PaginatedResult with data and has_more flag

        Note: Cursor pagination only supports single-column ordering for simplicity.
        Multi-column ordering is allowed without cursor but will raise an error with cursor.
        """
        pass

    async def fetch_one(
        self,
        table: str,
        where: Mapping[str, Any] | None = None,
        order_by: list[tuple[str, Literal["asc", "desc"]]] | None = None,
    ) -> dict[str, Any] | None:
        """
        Fetch one row from a table.
        """
        pass

    async def update(
        self,
        table: str,
        data: Mapping[str, Any],
        where: Mapping[str, Any],
    ) -> None:
        """
        Update a row in a table.
        """
        pass

    async def delete(
        self,
        table: str,
        where: Mapping[str, Any],
    ) -> None:
        """
        Delete a row from a table.
        """
        pass
