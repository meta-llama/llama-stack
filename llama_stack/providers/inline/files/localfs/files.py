# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import File, Form, Response, UploadFile

from llama_stack.apis.common.responses import Order
from llama_stack.apis.files import (
    Files,
    ListOpenAIFileResponse,
    OpenAIFileDeleteResponse,
    OpenAIFileObject,
    OpenAIFilePurpose,
)
from llama_stack.core.datatypes import AccessRule
from llama_stack.providers.utils.sqlstore.api import ColumnDefinition, ColumnType
from llama_stack.providers.utils.sqlstore.authorized_sqlstore import AuthorizedSqlStore
from llama_stack.providers.utils.sqlstore.sqlstore import sqlstore_impl

from .config import LocalfsFilesImplConfig


class LocalfsFilesImpl(Files):
    def __init__(self, config: LocalfsFilesImplConfig, policy: list[AccessRule]) -> None:
        self.config = config
        self.policy = policy
        self.sql_store: AuthorizedSqlStore | None = None

    async def initialize(self) -> None:
        """Initialize the files provider by setting up storage directory and metadata database."""
        # Create storage directory if it doesn't exist
        storage_path = Path(self.config.storage_dir)
        storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize SQL store for metadata
        self.sql_store = AuthorizedSqlStore(sqlstore_impl(self.config.metadata_store))
        await self.sql_store.create_table(
            "openai_files",
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "filename": ColumnType.STRING,
                "purpose": ColumnType.STRING,
                "bytes": ColumnType.INTEGER,
                "created_at": ColumnType.INTEGER,
                "expires_at": ColumnType.INTEGER,
                "file_path": ColumnType.STRING,  # Path to actual file on disk
            },
        )

    async def shutdown(self) -> None:
        pass

    def _generate_file_id(self) -> str:
        """Generate a unique file ID for OpenAI API."""
        return f"file-{uuid.uuid4().hex}"

    def _get_file_path(self, file_id: str) -> Path:
        """Get the filesystem path for a file ID."""
        return Path(self.config.storage_dir) / file_id

    # OpenAI Files API Implementation
    async def openai_upload_file(
        self,
        file: Annotated[UploadFile, File()],
        purpose: Annotated[OpenAIFilePurpose, Form()],
    ) -> OpenAIFileObject:
        """Upload a file that can be used across various endpoints."""
        if not self.sql_store:
            raise RuntimeError("Files provider not initialized")

        file_id = self._generate_file_id()
        file_path = self._get_file_path(file_id)

        content = await file.read()
        file_size = len(content)

        with open(file_path, "wb") as f:
            f.write(content)

        created_at = int(time.time())
        expires_at = created_at + self.config.ttl_secs

        await self.sql_store.insert(
            "openai_files",
            {
                "id": file_id,
                "filename": file.filename or "uploaded_file",
                "purpose": purpose.value,
                "bytes": file_size,
                "created_at": created_at,
                "expires_at": expires_at,
                "file_path": file_path.as_posix(),
            },
        )

        return OpenAIFileObject(
            id=file_id,
            filename=file.filename or "uploaded_file",
            purpose=purpose,
            bytes=file_size,
            created_at=created_at,
            expires_at=expires_at,
        )

    async def openai_list_files(
        self,
        after: str | None = None,
        limit: int | None = 10000,
        order: Order | None = Order.desc,
        purpose: OpenAIFilePurpose | None = None,
    ) -> ListOpenAIFileResponse:
        """Returns a list of files that belong to the user's organization."""
        if not self.sql_store:
            raise RuntimeError("Files provider not initialized")

        if not order:
            order = Order.desc

        where_conditions = {}
        if purpose:
            where_conditions["purpose"] = purpose.value

        paginated_result = await self.sql_store.fetch_all(
            table="openai_files",
            policy=self.policy,
            where=where_conditions if where_conditions else None,
            order_by=[("created_at", order.value)],
            cursor=("id", after) if after else None,
            limit=limit,
        )

        files = [
            OpenAIFileObject(
                id=row["id"],
                filename=row["filename"],
                purpose=OpenAIFilePurpose(row["purpose"]),
                bytes=row["bytes"],
                created_at=row["created_at"],
                expires_at=row["expires_at"],
            )
            for row in paginated_result.data
        ]

        return ListOpenAIFileResponse(
            data=files,
            has_more=paginated_result.has_more,
            first_id=files[0].id if files else "",
            last_id=files[-1].id if files else "",
        )

    async def openai_retrieve_file(self, file_id: str) -> OpenAIFileObject:
        """Returns information about a specific file."""
        if not self.sql_store:
            raise RuntimeError("Files provider not initialized")

        row = await self.sql_store.fetch_one("openai_files", policy=self.policy, where={"id": file_id})
        if not row:
            raise ValueError(f"File with id {file_id} not found")

        return OpenAIFileObject(
            id=row["id"],
            filename=row["filename"],
            purpose=OpenAIFilePurpose(row["purpose"]),
            bytes=row["bytes"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
        )

    async def openai_delete_file(self, file_id: str) -> OpenAIFileDeleteResponse:
        """Delete a file."""
        if not self.sql_store:
            raise RuntimeError("Files provider not initialized")

        row = await self.sql_store.fetch_one("openai_files", policy=self.policy, where={"id": file_id})
        if not row:
            raise ValueError(f"File with id {file_id} not found")

        # Delete physical file
        file_path = Path(row["file_path"])
        if file_path.exists():
            file_path.unlink()

        # Delete metadata from database
        await self.sql_store.delete("openai_files", where={"id": file_id})

        return OpenAIFileDeleteResponse(
            id=file_id,
            deleted=True,
        )

    async def openai_retrieve_file_content(self, file_id: str) -> Response:
        """Returns the contents of the specified file."""
        if not self.sql_store:
            raise RuntimeError("Files provider not initialized")

        # Get file metadata
        row = await self.sql_store.fetch_one("openai_files", policy=self.policy, where={"id": file_id})
        if not row:
            raise ValueError(f"File with id {file_id} not found")

        # Read file content
        file_path = Path(row["file_path"])
        if not file_path.exists():
            raise ValueError(f"File content not found on disk: {file_path}")

        with open(file_path, "rb") as f:
            content = f.read()

        # Return as binary response with appropriate content type
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{row["filename"]}"'},
        )
