# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field

from llama_stack.distribution.utils.config_dirs import RUNTIME_BASE_DIR
from llama_stack.providers.utils.kvstore.config import (
    KVStoreConfig,
    SqliteKVStoreConfig,
)


@json_schema_type
class PGVectorConfig(BaseModel):
    kvstore: KVStoreConfig = SqliteKVStoreConfig(
        db_path=(RUNTIME_BASE_DIR / "pgvector_store.db").as_posix()
    )  # Uses SQLite config specific to PGVector storage
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    db: str = Field(default="postgres")
    user: str = Field(default="postgres")
    password: str = Field(default="mysecretpassword")
