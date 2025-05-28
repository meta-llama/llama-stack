# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from llama_stack.distribution.utils.config_dirs import RUNTIME_BASE_DIR

from .api import SqlStore


class SqlStoreType(Enum):
    sqlite = "sqlite"
    postgres = "postgres"


class SqliteSqlStoreConfig(BaseModel):
    type: Literal["sqlite"] = SqlStoreType.sqlite.value
    db_path: str = Field(
        default=(RUNTIME_BASE_DIR / "sqlstore.db").as_posix(),
        description="Database path, e.g. ~/.llama/distributions/ollama/sqlstore.db",
    )

    @property
    def engine_str(self) -> str:
        return "sqlite+aiosqlite:///" + Path(self.db_path).expanduser().as_posix()

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, db_name: str = "sqlstore.db"):
        return cls(
            type="sqlite",
            db_path="${env.SQLITE_STORE_DIR:" + __distro_dir__ + "}/" + db_name,
        )

    # TODO: move this when we have a better way to specify dependencies with internal APIs
    @property
    def pip_packages(self) -> list[str]:
        return ["sqlalchemy[asyncio]"]


class PostgresSqlStoreConfig(BaseModel):
    type: Literal["postgres"] = SqlStoreType.postgres.value

    @property
    def pip_packages(self) -> list[str]:
        raise NotImplementedError("Postgres is not implemented yet")


SqlStoreConfig = Annotated[
    SqliteSqlStoreConfig | PostgresSqlStoreConfig,
    Field(discriminator="type", default=SqlStoreType.sqlite.value),
]


def sqlstore_impl(config: SqlStoreConfig) -> SqlStore:
    if config.type == SqlStoreType.sqlite.value:
        from .sqlite.sqlite import SqliteSqlStoreImpl

        impl = SqliteSqlStoreImpl(config)
    elif config.type == SqlStoreType.postgres.value:
        raise NotImplementedError("Postgres is not implemented yet")
    else:
        raise ValueError(f"Unknown sqlstore type {config.type}")

    return impl
