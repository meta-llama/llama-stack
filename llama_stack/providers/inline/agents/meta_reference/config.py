# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.utils.kvstore import KVStoreConfig
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig
from pydantic import BaseModel, Field


class MetaReferenceAgentsImplConfig(BaseModel):
    persistence_store: KVStoreConfig = Field(default=SqliteKVStoreConfig())
