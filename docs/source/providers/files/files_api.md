# Files API

## Overview

The Files API provides OpenAI-compatible file management capabilities for Llama Stack. It allows you to upload, store, retrieve, and manage files that can be used across various endpoints in your application.

## Features

- **File Upload**: Upload files with metadata and purpose classification
- **File Management**: List, retrieve, and delete files
- **Content Retrieval**: Access raw file content for processing
- **OpenAI Compatibility**: Full compatibility with OpenAI Files API endpoints
- **Flexible Storage**: Support for local filesystem and cloud storage backends

## API Endpoints

### Upload File

**POST** `/v1/openai/v1/files`

Upload a file that can be used across various endpoints.

**Request Body:**
- `file`: The file object to be uploaded (multipart form data)
- `purpose`: The intended purpose of the uploaded file

**Supported Purposes:**
- `assistants`: Files for use with assistants
- `fine-tune`: Files for fine-tuning models
- `batch`: Files for batch operations

**Response:**
```json
{
  "id": "file-abc123",
  "object": "file",
  "bytes": 140,
  "created_at": 1613779121,
  "filename": "mydata.jsonl",
  "purpose": "fine-tune"
}
```

**Example:**
```python
import requests

with open("data.jsonl", "rb") as f:
    files = {"file": f}
    data = {"purpose": "fine-tune"}
    response = requests.post(
        "http://localhost:8000/v1/openai/v1/files", files=files, data=data
    )
    file_info = response.json()
```

### List Files

**GET** `/v1/openai/v1/files`

Returns a list of files that belong to the user's organization.

**Query Parameters:**
- `after` (optional): A cursor for pagination
- `limit` (optional): Limit on number of objects (1-10,000, default: 10,000)
- `order` (optional): Sort order by created_at timestamp (`asc` or `desc`, default: `desc`)
- `purpose` (optional): Filter files by purpose

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "file-abc123",
      "object": "file",
      "bytes": 140,
      "created_at": 1613779121,
      "filename": "mydata.jsonl",
      "purpose": "fine-tune"
    }
  ],
  "has_more": false
}
```

**Example:**
```python
import requests

# List all files
response = requests.get("http://localhost:8000/v1/openai/v1/files")
files = response.json()

# List files with pagination
response = requests.get(
    "http://localhost:8000/v1/openai/v1/files",
    params={"limit": 10, "after": "file-abc123"},
)
files = response.json()

# Filter by purpose
response = requests.get(
    "http://localhost:8000/v1/openai/v1/files", params={"purpose": "fine-tune"}
)
files = response.json()
```

### Retrieve File

**GET** `/v1/openai/v1/files/{file_id}`

Returns information about a specific file.

**Path Parameters:**
- `file_id`: The ID of the file to retrieve

**Response:**
```json
{
  "id": "file-abc123",
  "object": "file",
  "bytes": 140,
  "created_at": 1613779121,
  "filename": "mydata.jsonl",
  "purpose": "fine-tune"
}
```

**Example:**
```python
import requests

file_id = "file-abc123"
response = requests.get(f"http://localhost:8000/v1/openai/v1/files/{file_id}")
file_info = response.json()
```

### Delete File

**DELETE** `/v1/openai/v1/files/{file_id}`

Delete a file.

**Path Parameters:**
- `file_id`: The ID of the file to delete

**Response:**
```json
{
  "id": "file-abc123",
  "object": "file",
  "deleted": true
}
```

**Example:**
```python
import requests

file_id = "file-abc123"
response = requests.delete(f"http://localhost:8000/v1/openai/v1/files/{file_id}")
result = response.json()
```

### Retrieve File Content

**GET** `/v1/openai/v1/files/{file_id}/content`

Returns the raw file content as a binary response.

**Path Parameters:**
- `file_id`: The ID of the file to retrieve content from

**Response:**
Binary file content with appropriate headers:
- `Content-Type`: `application/octet-stream`
- `Content-Disposition`: `attachment; filename="filename"`

**Example:**
```python
import requests

file_id = "file-abc123"
response = requests.get(f"http://localhost:8000/v1/openai/v1/files/{file_id}/content")

# Save content to file
with open("downloaded_file.jsonl", "wb") as f:
    f.write(response.content)

# Or process content directly
content = response.content
```

## Vector Store Integration

The Files API integrates with Vector Stores to enable document processing and search. For detailed information about this integration, see [OpenAI-Compatible File Operations and Vector Store Integration](../openai_file_operations_vector_stores.md).

### Vector Store File Operations

**List Vector Store Files:**
- **GET** `/v1/openai/v1/vector_stores/{vector_store_id}/files`

**Retrieve Vector Store File Content:**
- **GET** `/v1/openai/v1/vector_stores/{vector_store_id}/files/{file_id}/content`

**Attach File to Vector Store:**
- **POST** `/v1/openai/v1/vector_stores/{vector_store_id}/files`

## Error Handling

The Files API returns standard HTTP status codes and error responses:

- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: File not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

**Error Response Format:**
```json
{
  "error": {
    "message": "Error description",
    "type": "invalid_request_error",
    "code": "file_not_found"
  }
}
```

## Rate Limits

The Files API implements rate limiting to ensure fair usage:
- File uploads: 100 files per minute
- File retrievals: 1000 requests per minute
- File deletions: 100 requests per minute

## Best Practices

1. **File Organization**: Use descriptive filenames and appropriate purpose classifications
2. **Batch Operations**: For multiple files, consider using batch endpoints when available
3. **Error Handling**: Always check response status codes and handle errors gracefully
4. **Content Types**: Ensure files are uploaded with appropriate content types
5. **Cleanup**: Regularly delete unused files to manage storage costs

## Integration Examples

### With Python Client

```python
from llama_stack import LlamaStackClient

client = LlamaStackClient("http://localhost:8000")

# Upload a file
with open("data.jsonl", "rb") as f:
    file_info = await client.files.upload(file=f, purpose="fine-tune")

# List files
files = await client.files.list(purpose="fine-tune")

# Retrieve file content
content = await client.files.retrieve_content(file_info.id)
```

### With cURL

```bash
# Upload file
curl -X POST http://localhost:8000/v1/openai/v1/files \
  -F "file=@data.jsonl" \
  -F "purpose=fine-tune"

# List files
curl http://localhost:8000/v1/openai/v1/files

# Download file content
curl http://localhost:8000/v1/openai/v1/files/file-abc123/content \
  -o downloaded_file.jsonl
```

## Provider Support

The Files API supports multiple storage backends:

- **Local Filesystem**: Store files on local disk (inline provider)
- **S3**: Store files in AWS S3 or S3-compatible services (remote provider)
- **Custom Backends**: Extensible architecture for custom storage providers

See the [Files Providers](index.md) documentation for detailed configuration options.
