# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from typing import List, Optional, Protocol

from pydantic import BaseModel


class KVStoreValue(BaseModel):
    key: str
    value: str
    expiration: Optional[datetime] = None


class KVStore(Protocol):
    async def set(
        self, key: str, value: str, expiration: Optional[datetime] = None
    ) -> None: ...

    async def get(self, key: str) -> Optional[KVStoreValue]: ...

    async def delete(self, key: str) -> None: ...

    async def range(self, start_key: str, end_key: str) -> List[KVStoreValue]: ...
