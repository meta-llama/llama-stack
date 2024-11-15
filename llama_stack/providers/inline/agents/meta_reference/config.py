# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, Field

from llama_stack.providers.utils.kvstore import KVStoreConfig
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig


class MetaReferenceAgentsImplConfig(BaseModel):
    persistence_store: KVStoreConfig = Field(default=SqliteKVStoreConfig())

    @classmethod
    def sample_run_config(cls):
        return {
            "persistence_store": SqliteKVStoreConfig.sample_run_config(
                db_name="agents_store.db"
            ),
        }
