# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from datetime import datetime
from typing import Any, List, Optional

import aiosqlite

from llama_stack.apis.control_plane import *  # noqa: F403


from .config import SqliteControlPlaneConfig


class SqliteControlPlane(ControlPlane):
    def __init__(self, config: SqliteControlPlaneConfig):
        self.db_path = config.db_path
        self.table_name = config.table_name

    async def initialize(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    expiration TIMESTAMP
                )
            """
            )
            await db.commit()

    async def set(
        self, key: str, value: Any, expiration: Optional[datetime] = None
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f"INSERT OR REPLACE INTO {self.table_name} (key, value, expiration) VALUES (?, ?, ?)",
                (key, json.dumps(value), expiration),
            )
            await db.commit()

    async def get(self, key: str) -> Optional[ControlPlaneValue]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                f"SELECT value, expiration FROM {self.table_name} WHERE key = ?", (key,)
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                value, expiration = row
                return ControlPlaneValue(
                    key=key, value=json.loads(value), expiration=expiration
                )

    async def delete(self, key: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(f"DELETE FROM {self.table_name} WHERE key = ?", (key,))
            await db.commit()

    async def range(self, start_key: str, end_key: str) -> List[ControlPlaneValue]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                f"SELECT key, value, expiration FROM {self.table_name} WHERE key >= ? AND key <= ?",
                (start_key, end_key),
            ) as cursor:
                result = []
                async for row in cursor:
                    key, value, expiration = row
                    result.append(
                        ControlPlaneValue(
                            key=key, value=json.loads(value), expiration=expiration
                        )
                    )
                return result
