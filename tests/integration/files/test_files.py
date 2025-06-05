# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from io import BytesIO


def test_openai_client_basic_operations(openai_client):
    """Test basic file operations through OpenAI client."""
    client = openai_client

    test_content = b"files test content"

    try:
        # Upload file using OpenAI client
        with BytesIO(test_content) as file_buffer:
            file_buffer.name = "openai_test.txt"
            uploaded_file = client.files.create(file=file_buffer, purpose="assistants")

        # Verify basic response structure
        assert uploaded_file.id.startswith("file-")
        assert hasattr(uploaded_file, "filename")

        # List files
        files_list = client.files.list()
        file_ids = [f.id for f in files_list.data]
        assert uploaded_file.id in file_ids

        # Retrieve file info
        retrieved_file = client.files.retrieve(uploaded_file.id)
        assert retrieved_file.id == uploaded_file.id

        # Retrieve file content - OpenAI client returns httpx Response object
        content_response = client.files.content(uploaded_file.id)
        # The response is an httpx Response object with .content attribute containing bytes
        content = content_response.content
        assert content == test_content

        # Delete file
        delete_response = client.files.delete(uploaded_file.id)
        assert delete_response.deleted is True

    except Exception as e:
        # Cleanup in case of failure
        try:
            client.files.delete(uploaded_file.id)
        except Exception:
            pass
        raise e
