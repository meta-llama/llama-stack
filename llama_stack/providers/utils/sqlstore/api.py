# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Mapping
from enum import Enum
from typing import Any, Literal, Protocol

from pydantic import BaseModel


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
    ) -> list[dict[str, Any]]:
        """
        Fetch all rows from a table.
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
