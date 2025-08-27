# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time
import uuid
from typing import Annotated

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from fastapi import File, Form, Response, UploadFile

from llama_stack.apis.common.errors import ResourceNotFoundError
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

from .config import S3FilesImplConfig

# TODO: provider data for S3 credentials


def _create_s3_client(config: S3FilesImplConfig) -> boto3.client:
    try:
        s3_config = {
            "region_name": config.region,
        }

        # endpoint URL if specified (for MinIO, LocalStack, etc.)
        if config.endpoint_url:
            s3_config["endpoint_url"] = config.endpoint_url

        if config.aws_access_key_id and config.aws_secret_access_key:
            s3_config.update(
                {
                    "aws_access_key_id": config.aws_access_key_id,
                    "aws_secret_access_key": config.aws_secret_access_key,
                }
            )

        return boto3.client("s3", **s3_config)

    except (BotoCoreError, NoCredentialsError) as e:
        raise RuntimeError(f"Failed to initialize S3 client: {e}") from e


async def _create_bucket_if_not_exists(client: boto3.client, config: S3FilesImplConfig) -> None:
    try:
        client.head_bucket(Bucket=config.bucket_name)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            if not config.auto_create_bucket:
                raise RuntimeError(
                    f"S3 bucket '{config.bucket_name}' does not exist. "
                    f"Either create the bucket manually or set 'auto_create_bucket: true' in your configuration."
                ) from e
            try:
                # For us-east-1, we can't specify LocationConstraint
                if config.region == "us-east-1":
                    client.create_bucket(Bucket=config.bucket_name)
                else:
                    client.create_bucket(
                        Bucket=config.bucket_name,
                        CreateBucketConfiguration={"LocationConstraint": config.region},
                    )
            except ClientError as create_error:
                raise RuntimeError(
                    f"Failed to create S3 bucket '{config.bucket_name}': {create_error}"
                ) from create_error
        elif error_code == "403":
            raise RuntimeError(f"Access denied to S3 bucket '{config.bucket_name}'") from e
        else:
            raise RuntimeError(f"Failed to access S3 bucket '{config.bucket_name}': {e}") from e


class S3FilesImpl(Files):
    """S3-based implementation of the Files API."""

    # TODO: implement expiration, for now a silly offset
    _SILLY_EXPIRATION_OFFSET = 100 * 365 * 24 * 60 * 60

    def __init__(self, config: S3FilesImplConfig, policy: list[AccessRule]) -> None:
        self._config = config
        self.policy = policy
        self._client: boto3.client | None = None
        self._sql_store: AuthorizedSqlStore | None = None

    async def initialize(self) -> None:
        self._client = _create_s3_client(self._config)
        await _create_bucket_if_not_exists(self._client, self._config)

        self._sql_store = AuthorizedSqlStore(sqlstore_impl(self._config.metadata_store))
        await self._sql_store.create_table(
            "openai_files",
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "filename": ColumnType.STRING,
                "purpose": ColumnType.STRING,
                "bytes": ColumnType.INTEGER,
                "created_at": ColumnType.INTEGER,
                "expires_at": ColumnType.INTEGER,
                # TODO: add s3_etag field for integrity checking
            },
        )

    async def shutdown(self) -> None:
        pass

    @property
    def client(self) -> boto3.client:
        assert self._client is not None, "Provider not initialized"
        return self._client

    @property
    def sql_store(self) -> AuthorizedSqlStore:
        assert self._sql_store is not None, "Provider not initialized"
        return self._sql_store

    async def openai_upload_file(
        self,
        file: Annotated[UploadFile, File()],
        purpose: Annotated[OpenAIFilePurpose, Form()],
    ) -> OpenAIFileObject:
        file_id = f"file-{uuid.uuid4().hex}"

        filename = getattr(file, "filename", None) or "uploaded_file"

        created_at = int(time.time())
        expires_at = created_at + self._SILLY_EXPIRATION_OFFSET
        content = await file.read()
        file_size = len(content)

        await self.sql_store.insert(
            "openai_files",
            {
                "id": file_id,
                "filename": filename,
                "purpose": purpose.value,
                "bytes": file_size,
                "created_at": created_at,
                "expires_at": expires_at,
            },
        )

        try:
            self.client.put_object(
                Bucket=self._config.bucket_name,
                Key=file_id,
                Body=content,
                # TODO: enable server-side encryption
            )
        except ClientError as e:
            await self.sql_store.delete("openai_files", where={"id": file_id})

            raise RuntimeError(f"Failed to upload file to S3: {e}") from e

        return OpenAIFileObject(
            id=file_id,
            filename=filename,
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
        # this purely defensive. it should not happen because the router also default to Order.desc.
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
            # empty string or None? spec says str, ref impl returns str | None, we go with spec
            first_id=files[0].id if files else "",
            last_id=files[-1].id if files else "",
        )

    async def openai_retrieve_file(self, file_id: str) -> OpenAIFileObject:
        row = await self.sql_store.fetch_one("openai_files", policy=self.policy, where={"id": file_id})
        if not row:
            raise ResourceNotFoundError(file_id, "File", "files.list()")

        return OpenAIFileObject(
            id=row["id"],
            filename=row["filename"],
            purpose=OpenAIFilePurpose(row["purpose"]),
            bytes=row["bytes"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
        )

    async def openai_delete_file(self, file_id: str) -> OpenAIFileDeleteResponse:
        row = await self.sql_store.fetch_one("openai_files", policy=self.policy, where={"id": file_id})
        if not row:
            raise ResourceNotFoundError(file_id, "File", "files.list()")

        try:
            self.client.delete_object(
                Bucket=self._config.bucket_name,
                Key=row["id"],
            )
        except ClientError as e:
            if e.response["Error"]["Code"] != "NoSuchKey":
                raise RuntimeError(f"Failed to delete file from S3: {e}") from e

        await self.sql_store.delete("openai_files", where={"id": file_id})

        return OpenAIFileDeleteResponse(id=file_id, deleted=True)

    async def openai_retrieve_file_content(self, file_id: str) -> Response:
        row = await self.sql_store.fetch_one("openai_files", policy=self.policy, where={"id": file_id})
        if not row:
            raise ResourceNotFoundError(file_id, "File", "files.list()")

        try:
            response = self.client.get_object(
                Bucket=self._config.bucket_name,
                Key=row["id"],
            )
            # TODO: can we stream this instead of loading it into memory
            content = response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                await self.sql_store.delete("openai_files", where={"id": file_id})
                raise ResourceNotFoundError(file_id, "File", "files.list()") from e
            raise RuntimeError(f"Failed to download file from S3: {e}") from e

        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{row["filename"]}"'},
        )
