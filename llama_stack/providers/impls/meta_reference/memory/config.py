# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_models.schema_utils import json_schema_type
from llama_stack.providers.utils.kvstore.config import KVStoreConfig, SqliteKVStoreConfig
from pydantic import BaseModel


@json_schema_type
class FaissImplConfig(BaseModel):
    kvstore: KVStoreConfig = SqliteKVStoreConfig()  # Uses default SQLite config
