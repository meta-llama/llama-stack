import pytest

from llama_stack.providers.remote.files.object.s3.config import S3FilesImplConfig
from llama_stack.providers.remote.files.object.s3.s3_files import S3FilesAdapter


@pytest.fixture
def s3_config():
    return S3FilesImplConfig(
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
        region_name="us-east-1",
        endpoint_url="http://localhost:9000",
    )


@pytest.fixture
async def s3_files(s3_config):
    adapter = S3FilesAdapter(s3_config)
    await adapter.initialize()
    return adapter


@pytest.mark.asyncio
async def test_create_upload_session(s3_files):
    bucket = "test-bucket"
    key = "test-file.txt"
    mime_type = "text/plain"
    size = 1024

    response = await s3_files.create_upload_session(bucket, key, mime_type, size)
    assert response.id == f"{bucket}/{key}"
    assert response.size == size
    assert response.offset == 0
    assert response.url is not None
