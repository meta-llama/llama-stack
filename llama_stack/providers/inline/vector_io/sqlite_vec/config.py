# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# config.py
from typing import Any, Dict

from pydantic import BaseModel

from llama_stack.providers.utils.kvstore.config import (
    KVStoreConfig,
    SqliteKVStoreConfig,
)


class SQLiteVectorIOConfig(BaseModel):
    db_path: str
    kvstore: KVStoreConfig

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> Dict[str, Any]:
        return {
            "kvstore": SqliteKVStoreConfig.sample_run_config(
                __distro_dir__=__distro_dir__,
                db_name="sqlite_vec.db",
            )
        }
