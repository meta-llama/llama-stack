# OpenAI-Compatible File Operations and Vector Store Integration

## Overview

Llama Stack provides seamless integration between the Files API and Vector Store APIs, enabling you to upload documents and automatically process them into searchable vector embeddings. This integration implements OpenAI-compatible responses for file operations, following the [OpenAI Vector Store Files API specification](https://platform.openai.com/docs/api-reference/vector-stores-files).

## How It Works

The OpenAI-compatible file operations work through several key components:

1. **File Upload**: Documents are uploaded through the Files API
2. **Automatic Processing**: Files are automatically chunked and converted to embeddings
3. **Vector Storage**: Chunks are stored in vector databases with metadata
4. **Search & Retrieval**: Users can search through processed documents using natural language

## Supported Vector Store Providers

The following vector store providers support OpenAI-compatible file operations:

### Inline Providers (Single Node)

- **FAISS**: Fast in-memory vector similarity search
- **SQLite-vec**: Disk-based storage with hybrid search capabilities
- **Milvus**: High-performance vector database with advanced indexing

### Remote Providers (Hosted)

- **ChromaDB**: Vector database with metadata filtering
- **Qdrant**: Vector similarity search with payload filtering
- **Weaviate**: Vector database with GraphQL interface
- **Postgres (PGVector)**: Vector extensions for PostgreSQL

## File Processing Pipeline

### 1. File Upload

```python
from llama_stack import LlamaStackClient

client = LlamaStackClient("http://localhost:8000")

# Upload a document
with open("document.pdf", "rb") as f:
    file_info = await client.files.upload(file=f, purpose="assistants")
```

### 2. Attach to Vector Store

```python
# Create a vector store
vector_store = client.vector_stores.create(name="my_documents")

# Attach the file to the vector store
file_attach_response = await client.vector_stores.files.create(
    vector_store_id=vector_store.id, file_id=file_info.id
)
```

### 3. Automatic Processing

The system automatically:
- Detects the file type and extracts text content
- Splits content into optimal chunks (default: 800 tokens with 400 token overlap)
- Generates embeddings for each chunk
- Stores chunks with metadata in the vector store
- Updates file status to "completed"

### 4. Search and Retrieval

```python
# Search through processed documents
search_results = await client.vector_stores.search(
    vector_store_id=vector_store.id,
    query="What is the main topic discussed?",
    max_num_results=5,
)

# Process results
for result in search_results.data:
    print(f"Score: {result.score}")
    for content in result.content:
        print(f"Content: {content.text}")
```

## Supported File Types

The FileResponse system supports various document formats:

- **Text Files**: `.txt`, `.md`, `.rst`
- **Documents**: `.pdf`, `.docx`, `.doc`
- **Code**: `.py`, `.js`, `.java`, `.cpp`, etc.
- **Data**: `.json`, `.csv`, `.xml`
- **Web Content**: HTML files

## Chunking Strategies

### Default Strategy

The default chunking strategy uses:
- **Max Chunk Size**: 800 tokens
- **Overlap**: 400 tokens
- **Method**: Semantic boundary detection

### Custom Chunking

You can customize chunking when attaching files:

```python
from llama_stack.apis.vector_io import VectorStoreChunkingStrategy

# Custom chunking strategy
chunking_strategy = VectorStoreChunkingStrategy(
    type="custom", max_chunk_size_tokens=1000, chunk_overlap_tokens=200
)

# Attach file with custom chunking
file_attach_response = await client.vector_stores.files.create(
    vector_store_id=vector_store.id,
    file_id=file_info.id,
    chunking_strategy=chunking_strategy,
)
```

## File Management

### List Files in Vector Store

```python
# List all files in a vector store
files = await client.vector_stores.files.list(vector_store_id=vector_store.id)

for file in files:
    print(f"File: {file.filename}, Status: {file.status}")
```

### File Status Tracking

Files go through several statuses:
- **in_progress**: File is being processed
- **completed**: File successfully processed and searchable
- **failed**: Processing failed (check `last_error` for details)
- **cancelled**: Processing was cancelled

### Retrieve File Content

```python
# Get chunked content from vector store
content_response = await client.vector_stores.files.retrieve_content(
    vector_store_id=vector_store.id, file_id=file_info.id
)

for chunk in content_response.content:
    print(f"Chunk {chunk.metadata.get('chunk_index', 0)}: {chunk.text}")
```

## Search Capabilities

### Vector Search

Pure similarity search using embeddings:

```python
results = await client.vector_stores.search(
    vector_store_id=vector_store.id,
    query="machine learning algorithms",
    max_num_results=10,
)
```

### Filtered Search

Combine vector search with metadata filtering:

```python
results = await client.vector_stores.search(
    vector_store_id=vector_store.id,
    query="machine learning algorithms",
    filters={"file_type": "pdf", "upload_date": "2024-01-01"},
    max_num_results=10,
)
```

### Hybrid Search (SQLite-vec only)

SQLite-vec supports combining vector and keyword search:

```python
results = await client.vector_io.query_chunks(
    vector_db_id=vector_store.id,
    query="machine learning algorithms",
    params={
        "mode": "hybrid",
        "max_chunks": 10,
        "ranker": {"type": "rrf", "impact_factor": 60.0},
    },
)
```

## Performance Considerations

> **Note**: For detailed performance optimization strategies, see [Performance Considerations](../providers/files/openai_file_operations_support.md#performance-considerations) in the provider documentation.

**Key Points:**
- **Chunk Size**: 400-600 tokens for precision, 800-1200 for context
- **Storage**: Choose provider based on your performance needs
- **Search**: Optimize for your specific use case

## Error Handling

> **Note**: For comprehensive troubleshooting and error handling, see [Troubleshooting](../providers/files/openai_file_operations_support.md#troubleshooting) in the provider documentation.

**Common Issues:**
- File processing failures (format, size limits)
- Search performance optimization
- Storage and memory issues

## Best Practices

> **Note**: For detailed best practices and recommendations, see [Best Practices](../providers/files/openai_file_operations_support.md#best-practices) in the provider documentation.

**Key Recommendations:**
- File organization and naming conventions
- Chunking strategy optimization
- Metadata and monitoring practices
- Regular cleanup and maintenance

## Integration Examples

### RAG Application

```python
# Build a RAG system with file uploads
async def build_rag_system():
    # Create vector store
    vector_store = client.vector_stores.create(name="knowledge_base")

    # Upload and process documents
    documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    for doc in documents:
        with open(doc, "rb") as f:
            file_info = await client.files.upload(file=f, purpose="assistants")
            await client.vector_stores.files.create(
                vector_store_id=vector_store.id, file_id=file_info.id
            )

    return vector_store


# Query the RAG system
async def query_rag(vector_store_id, question):
    results = await client.vector_stores.search(
        vector_store_id=vector_store_id, query=question, max_num_results=5
    )
    return results
```

### Document Analysis

```python
# Analyze document content through vector search
async def analyze_document(vector_store_id, file_id):
    # Get document content
    content = await client.vector_stores.files.retrieve_content(
        vector_store_id=vector_store_id, file_id=file_id
    )

    # Search for specific topics
    topics = ["introduction", "methodology", "conclusion"]
    analysis = {}

    for topic in topics:
        results = await client.vector_stores.search(
            vector_store_id=vector_store_id, query=topic, max_num_results=3
        )
        analysis[topic] = results.data

    return analysis
```

## Next Steps

- Explore the [Files API documentation](../apis/files.md) for detailed API reference
- Check [Vector Store Providers](../providers/vector_io/index.md) for specific implementation details
- Review [Getting Started](../getting_started/index.md) for quick setup instructions
