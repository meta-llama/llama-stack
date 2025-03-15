# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel

from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol
from llama_stack.schema_utils import json_schema_type, webmethod


@json_schema_type
class FileUploadResponse(BaseModel):
    """
    Response after initiating a file upload session.

    :param id: ID of the upload session
    :param url: Upload URL for the file or file parts
    :param offset: Upload content offset
    :param size: Upload content size
    """

    id: str
    url: str
    offset: int
    size: int


@json_schema_type
class BucketResponse(BaseModel):
    name: str


@json_schema_type
class ListBucketResponse(BaseModel):
    """
    Response representing a list of file entries.

    :param data: List of FileResponse entries
    """

    data: List[BucketResponse]


@json_schema_type
class FileResponse(BaseModel):
    """
    Response representing a file entry.

    :param bucket: Bucket under which the file is stored (valid chars: a-zA-Z0-9_-)
    :param key: Key under which the file is stored (valid chars: a-zA-Z0-9_-/.)
    :param mime_type: MIME type of the file
    :param url: Upload URL for the file contents
    :param bytes: Size of the file in bytes
    :param created_at: Timestamp of when the file was created
    """

    bucket: str
    key: str
    mime_type: str
    url: str
    bytes: int
    created_at: int


@json_schema_type
class ListFileResponse(BaseModel):
    """
    Response representing a list of file entries.

    :param data: List of FileResponse entries
    """

    data: List[FileResponse]


@runtime_checkable
@trace_protocol
class Files(Protocol):
    @webmethod(route="/files", method="POST")
    async def create_upload_session(
        self,
        bucket: str,
        key: str,
        mime_type: str,
        size: int,
    ) -> FileUploadResponse:
        """
        Create a new upload session for a file identified by a bucket and key.

        :param bucket: Bucket under which the file is stored (valid chars: a-zA-Z0-9_-)
        :param key: Key under which the file is stored (valid chars: a-zA-Z0-9_-/.)
        :param mime_type: MIME type of the file
        :param size: File size in bytes
        """
        ...

    @webmethod(route="/files/session:{upload_id}", method="POST", raw_bytes_request_body=True)
    async def upload_content_to_session(
        self,
        upload_id: str,
    ) -> Optional[FileResponse]:
        """
        Upload file content to an existing upload session.
        On the server, request body will have the raw bytes that are uploaded.

        :param upload_id: ID of the upload session
        """
        ...

    @webmethod(route="/files/session:{upload_id}", method="GET")
    async def get_upload_session_info(
        self,
        upload_id: str,
    ) -> FileUploadResponse:
        """
        Returns information about an existsing upload session

        :param upload_id: ID of the upload session
        """
        ...

    @webmethod(route="/files", method="GET")
    async def list_all_buckets(
        self,
        bucket: str,
    ) -> ListBucketResponse:
        """
        List all buckets.
        """
        ...

    @webmethod(route="/files/{bucket}", method="GET")
    async def list_files_in_bucket(
        self,
        bucket: str,
    ) -> ListFileResponse:
        """
        List all files in a bucket.

        :param bucket: Bucket name (valid chars: a-zA-Z0-9_-)
        """
        ...

    @webmethod(route="/files/{bucket}/{key:path}", method="GET")
    async def get_file(
        self,
        bucket: str,
        key: str,
    ) -> FileResponse:
        """
        Get a file info identified by a bucket and key.

        :param bucket: Bucket name (valid chars: a-zA-Z0-9_-)
        :param key: Key under which the file is stored (valid chars: a-zA-Z0-9_-/.)
        """
        ...

    @webmethod(route="/files/{bucket}/{key:path}", method="DELETE")
    async def delete_file(
        self,
        bucket: str,
        key: str,
    ) -> FileResponse:
        """
        Delete a file identified by a bucket and key.

        :param bucket: Bucket name (valid chars: a-zA-Z0-9_-)
        :param key: Key under which the file is stored (valid chars: a-zA-Z0-9_-/.)
        """
        ...
