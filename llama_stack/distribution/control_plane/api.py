# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from typing import Any, List, Optional, Protocol

from llama_models.schema_utils import json_schema_type, webmethod
from pydantic import BaseModel


@json_schema_type
class ControlPlaneValue(BaseModel):
    key: str
    value: Any
    expiration: Optional[datetime] = None


@json_schema_type
class ControlPlane(Protocol):
    @webmethod(route="/control_plane/set")
    async def set(
        self, key: str, value: Any, expiration: Optional[datetime] = None
    ) -> None: ...

    @webmethod(route="/control_plane/get", method="GET")
    async def get(self, key: str) -> Optional[ControlPlaneValue]: ...

    @webmethod(route="/control_plane/delete")
    async def delete(self, key: str) -> None: ...

    @webmethod(route="/control_plane/range", method="GET")
    async def range(self, start_key: str, end_key: str) -> List[ControlPlaneValue]: ...
