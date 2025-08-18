# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Test suite for S3 Files Provider

This test suite covers the S3 implementation of the Llama Stack Files API, organized into the following categories:

CORE FILE OPERATIONS (Positive Tests):
- test_upload_file: Basic file upload functionality
- test_retrieve_file: File metadata retrieval
- test_retrieve_file_content: File content download with proper headers
- test_delete_file: File deletion with S3 backend verification
- test_upload_file_without_filename: Upload handling when filename is missing (edge case)

FILE LISTING AND FILTERING (Positive Tests):
- test_list_files_empty: Listing behavior when no files exist (boundary case)
- test_list_files: Listing multiple uploaded files
- test_list_files_with_purpose_filter: Filtering files by OpenAI purpose

ERROR HANDLING (Negative Tests):
- test_nonexistent_file_retrieval: Proper HTTP errors for missing file metadata
- test_nonexistent_file_content_retrieval: Proper HTTP errors for missing file content
- test_nonexistent_file_deletion: Proper HTTP errors for deleting non-existent files
- test_file_operations_when_s3_object_deleted: File operations when underlying S3 object is deleted

INFRASTRUCTURE:
- Uses moto library with mock_aws for S3 backend mocking
- Includes S3 backend verification to ensure actual S3 operations occur
- Tests use get_adapter_impl for realistic provider initialization
- Fixtures provide shared S3 client and proper test isolation

All tests verify both the Files API contract and the underlying S3 storage operations.
"""

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from llama_stack.apis.common.errors import ResourceNotFoundError
from llama_stack.apis.files import OpenAIFilePurpose
from llama_stack.providers.remote.files.s3 import (
    S3FilesImplConfig,
    get_adapter_impl,
)
from llama_stack.providers.utils.sqlstore.sqlstore import SqliteSqlStoreConfig


class MockUploadFile:
    def __init__(self, content: bytes, filename: str, content_type: str = "text/plain"):
        self.content = content
        self.filename = filename
        self.content_type = content_type

    async def read(self):
        return self.content


@pytest.fixture
def s3_config(tmp_path):
    db_path = tmp_path / "s3_files_metadata.db"

    return S3FilesImplConfig(
        bucket_name="test-bucket",
        region="not-a-region",
        metadata_store=SqliteSqlStoreConfig(db_path=db_path.as_posix()),
    )


@pytest.fixture
def s3_client():
    """Create a mocked S3 client for testing."""
    # we use `with mock_aws()` because @mock_aws decorator does not support being a generator
    with mock_aws():
        # must yield or the mock will be reset before it is used
        yield boto3.client("s3")


@pytest.fixture
async def s3_provider(s3_config, s3_client):
    """Create an S3 files provider with mocked S3 for testing."""
    provider = await get_adapter_impl(s3_config, {})
    yield provider
    await provider.shutdown()


@pytest.fixture
def sample_text_file():
    content = b"Hello, this is a test file for the S3 Files API!"
    return MockUploadFile(content, "sample_text_file.txt")


class TestS3FilesImpl:
    """Test suite for S3 Files implementation."""

    async def test_upload_file(self, s3_provider, sample_text_file, s3_client, s3_config):
        """Test successful file upload."""
        sample_text_file.filename = "test_upload_file"
        result = await s3_provider.openai_upload_file(
            file=sample_text_file,
            purpose=OpenAIFilePurpose.ASSISTANTS,
        )

        assert result.filename == sample_text_file.filename
        assert result.purpose == OpenAIFilePurpose.ASSISTANTS
        assert result.bytes == len(sample_text_file.content)
        assert result.id.startswith("file-")

        # Verify file exists in S3 backend
        response = s3_client.head_object(Bucket=s3_config.bucket_name, Key=result.id)
        assert response["ResponseMetadata"]["HTTPStatusCode"] == 200

    async def test_list_files_empty(self, s3_provider):
        """Test listing files when no files exist."""
        result = await s3_provider.openai_list_files()

        assert len(result.data) == 0
        assert not result.has_more
        assert result.first_id == ""
        assert result.last_id == ""

    async def test_retrieve_file(self, s3_provider, sample_text_file):
        """Test retrieving file metadata."""
        sample_text_file.filename = "test_retrieve_file"
        uploaded = await s3_provider.openai_upload_file(
            file=sample_text_file,
            purpose=OpenAIFilePurpose.ASSISTANTS,
        )

        retrieved = await s3_provider.openai_retrieve_file(uploaded.id)

        assert retrieved.id == uploaded.id
        assert retrieved.filename == uploaded.filename
        assert retrieved.purpose == uploaded.purpose
        assert retrieved.bytes == uploaded.bytes

    async def test_retrieve_file_content(self, s3_provider, sample_text_file):
        """Test retrieving file content."""
        sample_text_file.filename = "test_retrieve_file_content"
        uploaded = await s3_provider.openai_upload_file(
            file=sample_text_file,
            purpose=OpenAIFilePurpose.ASSISTANTS,
        )

        response = await s3_provider.openai_retrieve_file_content(uploaded.id)

        assert response.body == sample_text_file.content
        assert response.headers["Content-Disposition"] == f'attachment; filename="{sample_text_file.filename}"'

    async def test_delete_file(self, s3_provider, sample_text_file, s3_config, s3_client):
        """Test deleting a file."""
        sample_text_file.filename = "test_delete_file"
        uploaded = await s3_provider.openai_upload_file(
            file=sample_text_file,
            purpose=OpenAIFilePurpose.ASSISTANTS,
        )

        delete_response = await s3_provider.openai_delete_file(uploaded.id)

        assert delete_response.id == uploaded.id
        assert delete_response.deleted is True

        with pytest.raises(ResourceNotFoundError, match="not found"):
            await s3_provider.openai_retrieve_file(uploaded.id)

        # Verify file is gone from S3 backend
        with pytest.raises(ClientError) as exc_info:
            s3_client.head_object(Bucket=s3_config.bucket_name, Key=uploaded.id)
        assert exc_info.value.response["Error"]["Code"] == "404"

    async def test_list_files(self, s3_provider, sample_text_file):
        """Test listing files after uploading some."""
        sample_text_file.filename = "test_list_files_with_content_file1"
        file1 = await s3_provider.openai_upload_file(
            file=sample_text_file,
            purpose=OpenAIFilePurpose.ASSISTANTS,
        )

        file2_content = MockUploadFile(b"Second file content", "test_list_files_with_content_file2")
        file2 = await s3_provider.openai_upload_file(
            file=file2_content,
            purpose=OpenAIFilePurpose.BATCH,
        )

        result = await s3_provider.openai_list_files()

        assert len(result.data) == 2
        file_ids = {f.id for f in result.data}
        assert file1.id in file_ids
        assert file2.id in file_ids

    async def test_list_files_with_purpose_filter(self, s3_provider, sample_text_file):
        """Test listing files with purpose filter."""
        sample_text_file.filename = "test_list_files_with_purpose_filter_file1"
        file1 = await s3_provider.openai_upload_file(
            file=sample_text_file,
            purpose=OpenAIFilePurpose.ASSISTANTS,
        )

        file2_content = MockUploadFile(b"Batch file content", "test_list_files_with_purpose_filter_file2")
        await s3_provider.openai_upload_file(
            file=file2_content,
            purpose=OpenAIFilePurpose.BATCH,
        )

        result = await s3_provider.openai_list_files(purpose=OpenAIFilePurpose.ASSISTANTS)

        assert len(result.data) == 1
        assert result.data[0].id == file1.id
        assert result.data[0].purpose == OpenAIFilePurpose.ASSISTANTS

    async def test_nonexistent_file_retrieval(self, s3_provider):
        """Test retrieving a non-existent file raises error."""
        with pytest.raises(ResourceNotFoundError, match="not found"):
            await s3_provider.openai_retrieve_file("file-nonexistent")

    async def test_nonexistent_file_content_retrieval(self, s3_provider):
        """Test retrieving content of a non-existent file raises error."""
        with pytest.raises(ResourceNotFoundError, match="not found"):
            await s3_provider.openai_retrieve_file_content("file-nonexistent")

    async def test_nonexistent_file_deletion(self, s3_provider):
        """Test deleting a non-existent file raises error."""
        with pytest.raises(ResourceNotFoundError, match="not found"):
            await s3_provider.openai_delete_file("file-nonexistent")

    async def test_upload_file_without_filename(self, s3_provider, sample_text_file):
        """Test uploading a file without a filename uses the fallback."""
        del sample_text_file.filename
        result = await s3_provider.openai_upload_file(
            file=sample_text_file,
            purpose=OpenAIFilePurpose.ASSISTANTS,
        )

        assert result.purpose == OpenAIFilePurpose.ASSISTANTS
        assert result.bytes == len(sample_text_file.content)

        retrieved = await s3_provider.openai_retrieve_file(result.id)
        assert retrieved.filename == result.filename

    async def test_file_operations_when_s3_object_deleted(self, s3_provider, sample_text_file, s3_config, s3_client):
        """Test file operations when S3 object is deleted but metadata exists (negative test)."""
        sample_text_file.filename = "test_orphaned_metadata"
        uploaded = await s3_provider.openai_upload_file(
            file=sample_text_file,
            purpose=OpenAIFilePurpose.ASSISTANTS,
        )

        # Directly delete the S3 object from the backend
        s3_client.delete_object(Bucket=s3_config.bucket_name, Key=uploaded.id)

        retrieved_metadata_after = await s3_provider.openai_retrieve_file(uploaded.id)  # TODO: this should fail
        assert retrieved_metadata_after.id == uploaded.id

        listed_files = await s3_provider.openai_list_files()
        assert uploaded.id in [file.id for file in listed_files.data]

        with pytest.raises(ResourceNotFoundError, match="not found") as exc_info:
            await s3_provider.openai_retrieve_file_content(uploaded.id)

        assert "content" in str(exc_info).lower()
        assert uploaded.id in str(exc_info).lower()
