# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import aioboto3
import aiohttp
import botocore
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_file_upload_download_flow(s3_files, tmp_path):
    """Test complete file upload and download flow."""
    # Get the adapter from the fixture
    adapter = await anext(s3_files)

    # Test data
    bucket = "test-bucket"
    key = tmp_path / "test-file.txt"
    content = b"Hello, this is a test file content!"
    key.write_bytes(content)
    mime_type = "text/plain"

    # Create bucket and upload file
    async with aioboto3.Session().client(
        "s3",
        endpoint_url=adapter.config.endpoint_url,
        aws_access_key_id=adapter.config.aws_access_key_id,
        aws_secret_access_key=adapter.config.aws_secret_access_key,
        region_name=adapter.config.region_name,
    ) as s3:
        try:
            await s3.create_bucket(Bucket=bucket)
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "BucketAlreadyOwnedByYou":
                pass
            else:
                raise
        except Exception as e:
            print(f"Unexpected error creating bucket: {e}")
            raise

    # Create upload session
    upload_response = await adapter.create_upload_session(
        bucket=bucket, key=key.as_posix(), mime_type=mime_type, size=len(content)
    )

    # Upload content using the adapter
    response = await adapter.upload_content_to_session(upload_response.id)
    assert response is not None
    assert response.bucket == bucket
    assert response.key == str(key)
    assert response.bytes == len(content)

    # Verify file exists
    file_info = await adapter.get_file(bucket, key.as_posix())
    assert file_info.bucket == bucket
    assert file_info.key == key.as_posix()
    assert file_info.mime_type == mime_type
    assert file_info.bytes == len(content)

    # Download file using presigned URL
    async with aiohttp.ClientSession() as session:
        async with session.get(file_info.url) as response:
            assert response.status == 200
            downloaded_content = await response.read()
            assert downloaded_content == content

    # Clean up - delete the file
    await adapter.delete_file(bucket, key.as_posix())

    # Remove test bucket
    await s3.delete_bucket(Bucket=bucket)


@pytest.mark.asyncio
async def test_pagination(s3_files, tmp_path):
    """Test pagination functionality."""
    bucket = "pagination-test"
    files = [f"file_{i}.txt" for i in range(15)]
    content = b"test content"
    mime_type = "text/plain"

    # Get the adapter from the fixture
    adapter = await anext(s3_files)

    # Create bucket
    async with adapter.session.client(
        "s3",
        aws_access_key_id=adapter.config.aws_access_key_id,
        aws_secret_access_key=adapter.config.aws_secret_access_key,
        region_name=adapter.config.region_name,
        endpoint_url=adapter.config.endpoint_url,
    ) as s3:
        try:
            await s3.create_bucket(Bucket=bucket)
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "BucketAlreadyOwnedByYou":
                pass
            else:
                raise

    # Upload files using the proper upload methods
    for filename in files:
        # Create temporary file
        temp_file = tmp_path / filename
        temp_file.write_bytes(content)

        # Create upload session
        upload_response = await adapter.create_upload_session(
            bucket=bucket, key=filename, mime_type=mime_type, size=len(content)
        )

        # Upload content using the adapter
        response = await adapter.upload_content_to_session(upload_response.id)
        assert response is not None
        assert response.bucket == bucket
        assert response.key == filename
        assert response.bytes == len(content)

    # Test first page
    page1 = await adapter.list_files_in_bucket(bucket, page=1, size=5)
    assert len(page1.items) == 5
    assert page1.total == 15

    # Test second page
    page2 = await adapter.list_files_in_bucket(bucket, page=2, size=5)
    assert len(page2.items) == 5
    assert page2.total == 15

    # Verify no overlap between pages
    page1_keys = {item["key"] for item in page1.items}
    page2_keys = {item["key"] for item in page2.items}
    assert not page1_keys.intersection(page2_keys)

    # Also test list_all_buckets
    buckets = await adapter.list_all_buckets()
    assert len(buckets.data) > 0
    assert any(bucket["name"] == bucket for bucket in buckets.data)

    # Clean up - delete all files and the bucket
    async with adapter.session.client(
        "s3",
        aws_access_key_id=adapter.config.aws_access_key_id,
        aws_secret_access_key=adapter.config.aws_secret_access_key,
        region_name=adapter.config.region_name,
        endpoint_url=adapter.config.endpoint_url,
    ) as s3:
        for filename in files:
            await adapter.delete_file(bucket, filename)
        await s3.delete_bucket(Bucket=bucket)


# @pytest.mark.asyncio
# async def test_large_file_upload(s3_files):
#     """Test uploading a large file."""
#     bucket = "large-file-test"
#     key = "large-file.bin"
#     mime_type = "application/octet-stream"

#     # Create a 5MB file
#     content = os.urandom(5 * 1024 * 1024)

#     # Create bucket
#     async with s3_files.session.client("s3") as s3:
#         await s3.create_bucket(Bucket=bucket)

#     # Create upload session
#     upload_response = await s3_files.create_upload_session(
#         bucket=bucket, key=key, mime_type=mime_type, size=len(content)
#     )

#     # Upload content
#     async with aiohttp.ClientSession() as session:
#         async with session.put(upload_response.url, data=content) as response:
#             assert response.status == 200

#     # Verify file
#     file_info = await s3_files.get_file(bucket, key)
#     assert file_info.bytes == len(content)
#     assert file_info.mime_type == mime_type


# @pytest.mark.asyncio
# async def test_error_handling(s3_files):
#     """Test error handling for various scenarios."""
#     bucket = "error-test"
#     key = "non-existent.txt"

#     # Test getting non-existent file
#     with pytest.raises(Exception):
#         await s3_files.get_file(bucket, key)

#     # Test listing files in non-existent bucket
#     with pytest.raises(Exception):
#         await s3_files.list_files_in_bucket(bucket)

#     # Test deleting non-existent file
#     with pytest.raises(Exception):
#         await s3_files.delete_file(bucket, key)
