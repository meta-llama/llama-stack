# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import aioboto3
from botocore.exceptions import ClientError

from llama_stack.apis.common.responses import PaginatedResponse
from llama_stack.apis.files.files import (
    BucketResponse,
    FileResponse,
    Files,
    FileUploadResponse,
)
from llama_stack.log import get_logger
from llama_stack.providers.utils.kvstore import KVStore
from llama_stack.providers.utils.pagination import paginate_records

from .config import S3FilesImplConfig
from .persistence import S3FilesPersistence

logger = get_logger(name=__name__, category="files")


class S3FilesAdapter(Files):
    def __init__(self, config: S3FilesImplConfig, kvstore: KVStore):
        self.config = config
        self.session = aioboto3.Session()
        self.persistence = S3FilesPersistence(kvstore)

    async def initialize(self):
        # TODO: health check?
        pass

    async def create_upload_session(
        self,
        bucket: str,
        key: str,
        mime_type: str,
        size: int,
    ) -> FileUploadResponse:
        """Create a presigned URL for uploading a file to S3."""
        try:
            logger.debug(
                "create_upload_session",
                {"original_key": key, "s3_key": key, "bucket": bucket, "mime_type": mime_type, "size": size},
            )

            async with self.session.client(
                "s3",
                aws_access_key_id=self.config.aws_access_key_id,
                aws_secret_access_key=self.config.aws_secret_access_key,
                region_name=self.config.region_name,
                endpoint_url=self.config.endpoint_url,
            ) as s3:
                url = await s3.generate_presigned_url(
                    "put_object",
                    Params={
                        "Bucket": bucket,
                        "Key": key,
                        "ContentType": mime_type,
                    },
                    ExpiresIn=3600,  # URL expires in 1 hour - should it be longer?
                )
                logger.debug("Generated presigned URL", {"url": url})

                response = FileUploadResponse(
                    id=f"{bucket}/{key}",
                    url=url,
                    offset=0,
                    size=size,
                )

                # Store the session info
                await self.persistence.store_upload_session(
                    session_info=response,
                    bucket=bucket,
                    key=key,  # Store the original key for file reading
                    mime_type=mime_type,
                    size=size,
                )

                return response
        except ClientError as e:
            logger.error("S3 ClientError in create_upload_session", {"error": str(e)})
            raise Exception(f"Failed to create upload session: {str(e)}") from e

    async def upload_content_to_session(
        self,
        upload_id: str,
    ) -> FileResponse | None:
        """Upload content to S3 using the upload session."""

        try:
            # Get the upload session info from persistence
            session_info = await self.persistence.get_upload_session(upload_id)
            if not session_info:
                raise Exception(f"Upload session {upload_id} not found")

            logger.debug(
                "upload_content_to_session",
                {
                    "upload_id": upload_id,
                    "bucket": session_info.bucket,
                    "key": session_info.key,
                    "mime_type": session_info.mime_type,
                    "size": session_info.size,
                },
            )

            # Read the file content
            with open(session_info.key, "rb") as f:
                content = f.read()
            logger.debug("Read content", {"length": len(content)})

            # Use a single S3 client for all operations
            async with self.session.client(
                "s3",
                aws_access_key_id=self.config.aws_access_key_id,
                aws_secret_access_key=self.config.aws_secret_access_key,
                region_name=self.config.region_name,
                endpoint_url=self.config.endpoint_url,
            ) as s3:
                # Upload the content
                await s3.put_object(
                    Bucket=session_info.bucket, Key=session_info.key, Body=content, ContentType=session_info.mime_type
                )
                logger.debug("Upload successful")

                # Get the file info after upload
                response = await s3.head_object(Bucket=session_info.bucket, Key=session_info.key)
                logger.debug(
                    "File info retrieved",
                    {
                        "ContentType": response.get("ContentType"),
                        "ContentLength": response["ContentLength"],
                        "LastModified": response["LastModified"],
                    },
                )

                # Generate a presigned URL for reading
                url = await s3.generate_presigned_url(
                    "get_object",
                    Params={
                        "Bucket": session_info.bucket,
                        "Key": session_info.key,
                    },
                    ExpiresIn=3600,
                )

                return FileResponse(
                    bucket=session_info.bucket,
                    key=session_info.key,  # Use the original key to match test expectations
                    mime_type=response.get("ContentType", "application/octet-stream"),
                    url=url,
                    bytes=response["ContentLength"],
                    created_at=int(response["LastModified"].timestamp()),
                )
        except ClientError as e:
            logger.error("S3 ClientError in upload_content_to_session", {"error": str(e)})
            raise Exception(f"Failed to upload content: {str(e)}") from e
        finally:
            # Clean up the upload session
            await self.persistence.delete_upload_session(upload_id)

    async def get_upload_session_info(
        self,
        upload_id: str,
    ) -> FileUploadResponse:
        """Get information about an upload session."""
        bucket, key = upload_id.split("/", 1)
        try:
            async with self.session.client(
                "s3",
                aws_access_key_id=self.config.aws_access_key_id,
                aws_secret_access_key=self.config.aws_secret_access_key,
                region_name=self.config.region_name,
                endpoint_url=self.config.endpoint_url,
            ) as s3:
                response = await s3.head_object(Bucket=bucket, Key=key)
                url = await s3.generate_presigned_url(
                    "put_object",
                    Params={
                        "Bucket": bucket,
                        "Key": key,
                        "ContentType": response.get("ContentType", "application/octet-stream"),
                    },
                    ExpiresIn=3600,
                )
                return FileUploadResponse(
                    id=upload_id,
                    url=url,
                    offset=0,
                    size=response["ContentLength"],
                )
        except ClientError as e:
            raise Exception(f"Failed to get upload session info: {str(e)}") from e

    async def list_all_buckets(
        self,
        page: int | None = None,
        size: int | None = None,
    ) -> PaginatedResponse:
        """List all available S3 buckets."""

        try:
            response = await self.session.client(
                "s3",
                aws_access_key_id=self.config.aws_access_key_id,
                aws_secret_access_key=self.config.aws_secret_access_key,
                region_name=self.config.region_name,
                endpoint_url=self.config.endpoint_url,
            ).list_buckets()
            buckets = [BucketResponse(name=bucket["Name"]) for bucket in response["Buckets"]]
            # Convert BucketResponse objects to dictionaries for pagination
            bucket_dicts = [bucket.model_dump() for bucket in buckets]
            return paginate_records(bucket_dicts, page, size)
        except ClientError as e:
            raise Exception(f"Failed to list buckets: {str(e)}") from e

    async def list_files_in_bucket(
        self,
        bucket: str,
        page: int | None = None,
        size: int | None = None,
    ) -> PaginatedResponse:
        """List all files in an S3 bucket."""
        try:
            response = await self.session.client(
                "s3",
                aws_access_key_id=self.config.aws_access_key_id,
                aws_secret_access_key=self.config.aws_secret_access_key,
                region_name=self.config.region_name,
                endpoint_url=self.config.endpoint_url,
            ).list_objects_v2(Bucket=bucket)
            files: list[FileResponse] = []

            for obj in response.get("Contents", []):
                url = await self.session.client(
                    "s3",
                    aws_access_key_id=self.config.aws_access_key_id,
                    aws_secret_access_key=self.config.aws_secret_access_key,
                    region_name=self.config.region_name,
                    endpoint_url=self.config.endpoint_url,
                ).generate_presigned_url(
                    "get_object",
                    Params={
                        "Bucket": bucket,
                        "Key": obj["Key"],
                    },
                    ExpiresIn=3600,
                )

                files.append(
                    FileResponse(
                        bucket=bucket,
                        key=obj["Key"],
                        mime_type="application/octet-stream",  # Default mime type
                        url=url,
                        bytes=obj["Size"],
                        created_at=int(obj["LastModified"].timestamp()),
                    )
                )

            # Convert FileResponse objects to dictionaries for pagination
            file_dicts = [file.model_dump() for file in files]
            return paginate_records(file_dicts, page, size)
        except ClientError as e:
            raise Exception(f"Failed to list files in bucket: {str(e)}") from e

    async def get_file(
        self,
        bucket: str,
        key: str,
    ) -> FileResponse:
        """Get information about a specific file in S3."""
        try:
            async with self.session.client(
                "s3",
                aws_access_key_id=self.config.aws_access_key_id,
                aws_secret_access_key=self.config.aws_secret_access_key,
                region_name=self.config.region_name,
                endpoint_url=self.config.endpoint_url,
            ) as s3:
                response = await s3.head_object(Bucket=bucket, Key=key)
                url = await s3.generate_presigned_url(
                    "get_object",
                    Params={
                        "Bucket": bucket,
                        "Key": key,
                    },
                    ExpiresIn=3600,
                )

                return FileResponse(
                    bucket=bucket,
                    key=key,
                    mime_type=response.get("ContentType", "application/octet-stream"),
                    url=url,
                    bytes=response["ContentLength"],
                    created_at=int(response["LastModified"].timestamp()),
                )
        except ClientError as e:
            raise Exception(f"Failed to get file info: {str(e)}") from e

    async def delete_file(
        self,
        bucket: str,
        key: str,
    ) -> None:
        """Delete a file from S3."""
        try:
            async with self.session.client(
                "s3",
                aws_access_key_id=self.config.aws_access_key_id,
                aws_secret_access_key=self.config.aws_secret_access_key,
                region_name=self.config.region_name,
                endpoint_url=self.config.endpoint_url,
            ) as s3:
                await s3.delete_object(Bucket=bucket, Key=key)
        except ClientError as e:
            raise Exception(f"Failed to delete file: {str(e)}") from e
