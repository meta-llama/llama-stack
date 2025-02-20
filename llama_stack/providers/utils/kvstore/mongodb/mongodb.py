# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
from datetime import datetime
from typing import List, Optional

from pymongo import MongoClient

from llama_stack.providers.utils.kvstore import KVStore, MongoDBKVStoreConfig

log = logging.getLogger(__name__)


class MongoDBKVStoreImpl(KVStore):
    def __init__(self, config: MongoDBKVStoreConfig):
        self.config = config
        self.conn = None
        self.collection = None

    async def initialize(self) -> None:
        try:
            conn_creds = {
                "host": self.config.host,
                "port": self.config.port,
                "username": self.config.user,
                "password": self.config.password,
            }
            conn_creds = {k: v for k, v in conn_creds.items() if v is not None}
            self.conn = MongoClient(**conn_creds)
            self.collection = self.conn[self.config.db][self.config.collection_name]
        except Exception as e:
            log.exception("Could not connect to MongoDB database server")
            raise RuntimeError("Could not connect to MongoDB database server") from e

    def _namespaced_key(self, key: str) -> str:
        if not self.config.namespace:
            return key
        return f"{self.config.namespace}:{key}"

    async def set(
        self, key: str, value: str, expiration: Optional[datetime] = None
    ) -> None:

        key = self._namespaced_key(key)
        update_query = {"$set": {"value": value, "expiration": expiration}}
        self.collection.update_one({"key": key}, update_query, upsert=True)

    async def get(self, key: str) -> Optional[str]:
        key = self._namespaced_key(key)
        query = {"key": key}
        result = self.collection.find_one(query, {"value": 1, "_id": 0})
        return result["value"] if result else None

    async def delete(self, key: str) -> None:
        key = self._namespaced_key(key)
        self.collection.delete_one({"key": key})

    async def range(self, start_key: str, end_key: str) -> List[str]:
        start_key = self._namespaced_key(start_key)
        end_key = self._namespaced_key(end_key)
        query = {
            "key": {"$gte": start_key, "$lt": end_key},
        }
        cursor = self.collection.find(query, {"value": 1, "_id": 0}).sort("key", 1)
        return [doc["value"] for doc in cursor]
