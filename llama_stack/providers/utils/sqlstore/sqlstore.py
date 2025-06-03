# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from llama_stack.distribution.utils.config_dirs import RUNTIME_BASE_DIR

from .api import SqlStore

sql_store_pip_packages = ["sqlalchemy[asyncio]", "aiosqlite", "asyncpg"]


class SqlStoreType(Enum):
    sqlite = "sqlite"
    postgres = "postgres"


class SqlAlchemySqlStoreConfig(BaseModel):
    @property
    @abstractmethod
    def engine_str(self) -> str: ...

    # TODO: move this when we have a better way to specify dependencies with internal APIs
    @property
    def pip_packages(self) -> list[str]:
        return ["sqlalchemy[asyncio]"]


class SqliteSqlStoreConfig(SqlAlchemySqlStoreConfig):
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

    @property
    def pip_packages(self) -> list[str]:
        return super().pip_packages + ["aiosqlite"]


class PostgresSqlStoreConfig(SqlAlchemySqlStoreConfig):
    type: Literal["postgres"] = SqlStoreType.postgres.value
    host: str = "localhost"
    port: str = "5432"
    db: str = "llamastack"
    user: str
    password: str | None = None

    @property
    def engine_str(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

    @property
    def pip_packages(self) -> list[str]:
        return super().pip_packages + ["asyncpg"]

    @classmethod
    def sample_run_config(cls, **kwargs):
        return cls(
            type="postgres",
            host="${env.POSTGRES_HOST:localhost}",
            port="${env.POSTGRES_PORT:5432}",
            db="${env.POSTGRES_DB:llamastack}",
            user="${env.POSTGRES_USER:llamastack}",
            password="${env.POSTGRES_PASSWORD:llamastack}",
        )


SqlStoreConfig = Annotated[
    SqliteSqlStoreConfig | PostgresSqlStoreConfig,
    Field(discriminator="type", default=SqlStoreType.sqlite.value),
]


def sqlstore_impl(config: SqlStoreConfig) -> SqlStore:
    if config.type in [SqlStoreType.sqlite.value, SqlStoreType.postgres.value]:
        from .sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl

        impl = SqlAlchemySqlStoreImpl(config)
    else:
        raise ValueError(f"Unknown sqlstore type {config.type}")

    return impl
