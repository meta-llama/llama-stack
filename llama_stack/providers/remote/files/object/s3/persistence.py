# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import logging
from datetime import datetime, timezone

from pydantic import BaseModel

from llama_stack.apis.files.files import FileUploadResponse
from llama_stack.providers.utils.kvstore import KVStore

log = logging.getLogger(__name__)


class UploadSessionInfo(BaseModel):
    """Information about an upload session."""

    upload_id: str
    bucket: str
    key: str  # Original key for file reading
    s3_key: str  # S3 key for S3 operations
    mime_type: str
    size: int
    url: str
    created_at: datetime


class S3FilesPersistence:
    def __init__(self, kvstore: KVStore):
        self._kvstore = kvstore
        self._store: KVStore | None = None

    async def _get_store(self) -> KVStore:
        """Get the kvstore instance, initializing it if needed."""
        if self._store is None:
            self._store = self._kvstore
        return self._store

    async def store_upload_session(
        self, session_info: FileUploadResponse, bucket: str, key: str, mime_type: str, size: int
    ):
        """Store upload session information."""
        upload_info = UploadSessionInfo(
            upload_id=session_info.id,
            bucket=bucket,
            key=key,
            s3_key=key,
            mime_type=mime_type,
            size=size,
            url=session_info.url,
            created_at=datetime.now(timezone.utc),
        )

        store = await self._get_store()
        await store.set(
            key=f"upload_session:{session_info.id}",
            value=upload_info.model_dump_json(),
        )

    async def get_upload_session(self, upload_id: str) -> UploadSessionInfo | None:
        """Get upload session information."""
        store = await self._get_store()
        value = await store.get(
            key=f"upload_session:{upload_id}",
        )
        if not value:
            return None

        return UploadSessionInfo(**json.loads(value))

    async def delete_upload_session(self, upload_id: str) -> None:
        """Delete upload session information."""
        store = await self._get_store()
        await store.delete(key=f"upload_session:{upload_id}")
