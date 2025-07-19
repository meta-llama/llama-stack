# Vector IO Embedding Model Configuration

This guide explains how to configure embedding models for vector IO providers in Llama Stack, enabling you to use different embedding models for different use cases and optimize performance and storage requirements.

## Overview

Vector IO providers now support configurable embedding models at the provider level. This allows you to:

- **Use different embedding models** for different vector databases based on your use case
- **Optimize for performance** with lightweight models for fast retrieval
- **Optimize for quality** with high-dimensional models for semantic search
- **Save storage space** with variable-dimension embeddings (Matryoshka embeddings)
- **Ensure consistency** with provider-level defaults

## Configuration Options

Each vector IO provider configuration can include:

- `embedding_model`: The default embedding model ID to use for this provider
- `embedding_dimension`: Optional dimension override for models with variable dimensions

## Priority Order

The system uses the following priority order for embedding model selection:

1. **Explicit API parameters** (highest priority)
2. **Provider configuration defaults** (new feature)
3. **System default** from model registry (fallback)

## Example Configurations

### Fast Local Search with Lightweight Embeddings

```yaml
vector_io:
  - provider_id: fast_search
    provider_type: inline::faiss
    config:
      db_path: ~/.llama/faiss_fast.db
      embedding_model: "all-MiniLM-L6-v2"  # Fast, 384-dimensional
      embedding_dimension: 384
```

### High-Quality Semantic Search

```yaml
vector_io:
  - provider_id: quality_search
    provider_type: inline::sqlite_vec
    config:
      db_path: ~/.llama/sqlite_quality.db
      embedding_model: "sentence-transformers/all-mpnet-base-v2"  # High quality, 768-dimensional
      embedding_dimension: 768
```

### Storage-Optimized with Matryoshka Embeddings

```yaml
vector_io:
  - provider_id: compact_search
    provider_type: inline::faiss
    config:
      db_path: ~/.llama/faiss_compact.db
      embedding_model: "nomic-embed-text"  # Matryoshka model
      embedding_dimension: 256  # Reduced from default 768 for storage efficiency
```

### Cloud Deployment with OpenAI Embeddings

```yaml
vector_io:
  - provider_id: cloud_search
    provider_type: remote::qdrant
    config:
      api_key: "${env.QDRANT_API_KEY}"
      url: "${env.QDRANT_URL}"
      embedding_model: "text-embedding-3-small"
      embedding_dimension: 1536
```

## Model Registry Setup

Ensure your embedding models are properly configured in the model registry:

```yaml
models:
  # Lightweight model
  - model_id: all-MiniLM-L6-v2
    provider_id: local_inference
    provider_model_id: sentence-transformers/all-MiniLM-L6-v2
    model_type: embedding
    metadata:
      embedding_dimension: 384
      description: "Fast, lightweight embeddings"

  # High-quality model
  - model_id: sentence-transformers/all-mpnet-base-v2
    provider_id: local_inference
    provider_model_id: sentence-transformers/all-mpnet-base-v2
    model_type: embedding
    metadata:
      embedding_dimension: 768
      description: "High-quality embeddings"

  # Matryoshka model
  - model_id: nomic-embed-text
    provider_id: local_inference
    provider_model_id: nomic-embed-text
    model_type: embedding
    metadata:
      embedding_dimension: 768  # Default dimension
      description: "Variable-dimension Matryoshka embeddings"
```

## Use Cases

### Multi-Environment Setup

Configure different providers for different environments:

```yaml
vector_io:
  # Development - fast, lightweight
  - provider_id: dev_search
    provider_type: inline::faiss
    config:
      db_path: ~/.llama/dev_faiss.db
      embedding_model: "all-MiniLM-L6-v2"
      embedding_dimension: 384

  # Production - high quality, scalable
  - provider_id: prod_search
    provider_type: remote::qdrant
    config:
      api_key: "${env.QDRANT_API_KEY}"
      embedding_model: "text-embedding-3-large"
      embedding_dimension: 3072
```

### Domain-Specific Models

Use different models for different content types:

```yaml
vector_io:
  # Code search - specialized model
  - provider_id: code_search
    provider_type: inline::sqlite_vec
    config:
      db_path: ~/.llama/code_vectors.db
      embedding_model: "microsoft/codebert-base"
      embedding_dimension: 768

  # General documents - general-purpose model
  - provider_id: doc_search
    provider_type: inline::sqlite_vec
    config:
      db_path: ~/.llama/doc_vectors.db
      embedding_model: "all-mpnet-base-v2"
      embedding_dimension: 768
```

## Backward Compatibility

If no embedding model is specified in the provider configuration, the system will fall back to the existing behavior of using the first available embedding model from the model registry.

## Supported Providers

The configurable embedding models feature is supported by:

- **Inline providers**: Faiss, SQLite-vec, Milvus, ChromaDB, Qdrant
- **Remote providers**: Qdrant, Milvus, ChromaDB, PGVector, Weaviate

## Best Practices

1. **Match dimensions**: Ensure `embedding_dimension` matches your model's output
2. **Use variable dimensions wisely**: Only override dimensions for Matryoshka models that support it
3. **Consider performance trade-offs**: Smaller dimensions = faster search, larger = better quality
4. **Test configurations**: Validate your setup with sample queries before production use
5. **Document your choices**: Comment your configurations to explain model selection rationale