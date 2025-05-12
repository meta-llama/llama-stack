# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from llama_stack.apis.common.responses import PaginatedResponse
from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol
from llama_stack.schema_utils import json_schema_type, webmethod


@json_schema_type
class ArtifactUploadResponse(BaseModel):
    """
    Response after initiating an artifact upload session.

    :param id: ID of the upload session
    :param uri: Upload URI for the artifact or artifact parts
    :param offset: Upload content offset
    :param size: Upload content size
    """

    id: str
    uri: str
    offset: int
    size: int


@json_schema_type
class ArtifactResponse(BaseModel):
    """
    Response representing an artifact entry.

    :param key: Key under which the artifact is stored (valid chars: a-zA-Z0-9_-/.)
    :param mime_type: MIME type of the artifact
    :param uri: Upload URI for the artifact contents
    :param size: Size of the artifact in bytes
    :param created_at: Timestamp of when the artifact was created
    """

    key: str
    mime_type: str
    uri: str
    size: int
    created_at: int


@json_schema_type
class ArtifactDeleteRequest(BaseModel):
    """
    Request model for deleting a single artifact.

    :param key: Key under which the artifact is stored (valid chars: a-zA-Z0-9_-/.)
    """

    key: str


@json_schema_type
class BulkDeleteRequest(BaseModel):
    """
    Request model for bulk deletion of artifacts.

    :param artifacts: List of artifacts to delete
    """

    artifacts: list[ArtifactDeleteRequest]


@runtime_checkable
@trace_protocol
class Artifacts(Protocol):
    @webmethod(route="/artifacts", method="POST")
    async def create_upload_session(
        self,
        key: str,
        mime_type: str,
        size: int,
    ) -> ArtifactUploadResponse:
        """
        Create a new upload session for an artifact identified by a key.

        :param key: Key under which the artifact is stored (valid chars: a-zA-Z0-9_-/.)
        :param mime_type: MIME type of the artifact
        :param size: Artifact size in bytes
        """
        ...

    @webmethod(route="/artifacts/sessions/{upload_id}", method="POST", raw_bytes_request_body=True)
    async def upload_content_to_session(
        self,
        upload_id: str,
    ) -> ArtifactResponse | None:
        """
        Upload artifact content to an existing upload session.
        On the server, request body will have the raw bytes that are uploaded.

        :param upload_id: ID of the upload session
        """
        ...

    @webmethod(route="/artifacts/sessions/{upload_id}", method="GET")
    async def get_upload_session_info(
        self,
        upload_id: str,
    ) -> ArtifactUploadResponse:
        """
        Returns information about an existing upload session

        :param upload_id: ID of the upload session
        """
        ...

    @webmethod(route="/artifacts", method="GET")
    async def list_artifacts(
        self,
        start_index: int | None = None,
        limit: int | None = None,
    ) -> PaginatedResponse:
        """
        List all artifacts with pagination.

        :param start_index: Start index of the artifacts to list
        :param limit: Number of artifacts to list
        """
        ...

    @webmethod(route="/artifacts/{key:path}", method="GET")
    async def get_artifact(
        self,
        key: str,
    ) -> ArtifactResponse:
        """
        Get an artifact info identified by a key.

        :param key: Key under which the artifact is stored (valid chars: a-zA-Z0-9_-/.)
        """
        ...

    @webmethod(route="/artifacts/{key:path}", method="DELETE")
    async def delete_artifact(
        self,
        key: str,
    ) -> None:
        """
        Delete an artifact identified by a key.

        :param key: Key under which the artifact is stored (valid chars: a-zA-Z0-9_-/.)
        """
        ...

    @webmethod(route="/artifacts/bulk", method="POST")
    async def bulk_delete_artifacts(
        self,
        request: BulkDeleteRequest,
    ) -> None:
        """
        Delete multiple artifacts in a single request.

        :param request: Bulk delete request containing list of artifacts to delete
        """
        ...
