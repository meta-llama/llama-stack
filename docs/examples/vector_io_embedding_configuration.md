# Vector IO Embedding Model Configuration

## Overview

Vector IO providers now support configuring default embedding models at the provider level. This allows you to:
- Set a default embedding model for each vector store provider
- Support Matryoshka embeddings with custom dimensions
- Automatic dimension lookup from the model registry
- Maintain backward compatibility with existing configurations

## Configuration Options

### Provider-Level Embedding Configuration

Add `embedding_model` and `embedding_dimension` fields to your vector IO provider configuration:

```yaml
providers:
  vector_io:
    - provider_id: my_faiss_store
      provider_type: inline::faiss
      config:
        kvstore:
          provider_type: sqlite
          config:
            db_path: ~/.llama/distributions/my-app/faiss_store.db
        # NEW: Configure default embedding model
        embedding_model: "all-MiniLM-L6-v2"
        # Optional: Only needed for variable-dimension models
        # embedding_dimension: 384
```

### Embedding Model Selection Priority

The system uses a 3-tier priority system for selecting embedding models:

1. **Explicit API Parameters** (highest priority)
   ```python
   # API call explicitly specifies model - this takes precedence
   await vector_io.openai_create_vector_store(
       name="my-store",
       embedding_model="nomic-embed-text",  # Explicit override
       embedding_dimension=256,
   )
   ```

2. **Provider Config Defaults** (middle priority)
   ```yaml
   # Provider config provides default when no explicit model specified
   config:
     embedding_model: "all-MiniLM-L6-v2"
     embedding_dimension: 384
   ```

3. **System Default** (fallback)
   ```
   # Uses first available embedding model from model registry
   # Maintains backward compatibility
   ```

## Provider Examples

### FAISS with Default Embedding Model

```yaml
providers:
  vector_io:
    - provider_id: faiss_store
      provider_type: inline::faiss
      config:
        kvstore:
          provider_type: sqlite
          config:
            db_path: ~/.llama/distributions/my-app/faiss_store.db
        embedding_model: "all-MiniLM-L6-v2"
        # Dimension auto-lookup: 384 (from model registry)
```

### SQLite Vec with Matryoshka Embedding

```yaml
providers:
  vector_io:
    - provider_id: sqlite_vec_store
      provider_type: inline::sqlite_vec
      config:
        db_path: ~/.llama/distributions/my-app/sqlite_vec.db
        kvstore:
          provider_type: sqlite
          config:
            db_name: sqlite_vec_registry.db
        embedding_model: "nomic-embed-text"
        embedding_dimension: 256  # Override default 768 to 256
```

### Chroma with Provider Default

```yaml
providers:
  vector_io:
    - provider_id: chroma_store
      provider_type: inline::chroma
      config:
        db_path: ~/.llama/distributions/my-app/chroma.db
        embedding_model: "sentence-transformers/all-mpnet-base-v2"
        # Auto-lookup dimension from model registry
```

### Remote Qdrant Configuration

```yaml
providers:
  vector_io:
    - provider_id: qdrant_cloud
      provider_type: remote::qdrant
      config:
        api_key: "${env.QDRANT_API_KEY}"
        url: "https://my-cluster.qdrant.tech"
        embedding_model: "text-embedding-3-small"
        embedding_dimension: 512  # Custom dimension for Matryoshka model
```

### Multiple Providers with Different Models

```yaml
providers:
  vector_io:
    # Fast, lightweight embeddings for simple search
    - provider_id: fast_search
      provider_type: inline::faiss
      config:
        kvstore:
          provider_type: sqlite
          config:
            db_path: ~/.llama/fast_search.db
        embedding_model: "all-MiniLM-L6-v2"  # 384 dimensions

    # High-quality embeddings for semantic search
    - provider_id: semantic_search
      provider_type: remote::qdrant
      config:
        api_key: "${env.QDRANT_API_KEY}"
        embedding_model: "text-embedding-3-large"  # 3072 dimensions

    # Flexible Matryoshka embeddings
    - provider_id: flexible_search
      provider_type: inline::chroma
      config:
        db_path: ~/.llama/flexible_search.db
        embedding_model: "nomic-embed-text"
        embedding_dimension: 256  # Reduced from default 768
```

## Model Registry Configuration

Ensure your embedding models are registered in the model registry:

```yaml
models:
  - model_id: all-MiniLM-L6-v2
    provider_id: huggingface
    provider_model_id: sentence-transformers/all-MiniLM-L6-v2
    model_type: embedding
    metadata:
      embedding_dimension: 384

  - model_id: nomic-embed-text
    provider_id: ollama
    provider_model_id: nomic-embed-text
    model_type: embedding
    metadata:
      embedding_dimension: 768  # Default, can be overridden

  - model_id: text-embedding-3-small
    provider_id: openai
    provider_model_id: text-embedding-3-small
    model_type: embedding
    metadata:
      embedding_dimension: 1536  # Default for OpenAI model
```

## API Usage Examples

### Using Provider Defaults

```python
# Uses the embedding model configured in the provider config
vector_store = await vector_io.openai_create_vector_store(
    name="documents", provider_id="faiss_store"  # Will use configured embedding_model
)
```

### Explicit Override

```python
# Overrides provider defaults with explicit parameters
vector_store = await vector_io.openai_create_vector_store(
    name="documents",
    embedding_model="text-embedding-3-large",  # Override provider default
    embedding_dimension=1024,  # Custom dimension
    provider_id="faiss_store",
)
```

### Matryoshka Embedding Usage

```python
# Provider configured with nomic-embed-text and dimension 256
vector_store = await vector_io.openai_create_vector_store(
    name="compact_embeddings", provider_id="flexible_search"  # Uses Matryoshka config
)

# Or override with different dimension
vector_store = await vector_io.openai_create_vector_store(
    name="full_embeddings",
    embedding_dimension=768,  # Use full dimension
    provider_id="flexible_search",
)
```

## Migration Guide

### Updating Existing Configurations

Your existing configurations will continue to work without changes. To add provider-level defaults:

1. **Add embedding model fields** to your provider configs
2. **Test the configuration** to ensure expected behavior
3. **Remove explicit embedding_model parameters** from API calls if desired

### Before (explicit parameters required):
```python
# Had to specify embedding model every time
await vector_io.openai_create_vector_store(
    name="store1", embedding_model="all-MiniLM-L6-v2"
)
```

### After (provider defaults):
```yaml
# Configure once in provider config
config:
  embedding_model: "all-MiniLM-L6-v2"
```

```python
# No need to specify repeatedly
await vector_io.openai_create_vector_store(name="store1")
await vector_io.openai_create_vector_store(name="store2")
await vector_io.openai_create_vector_store(name="store3")
```

## Best Practices

### 1. Model Selection
- Use **lightweight models** (e.g., `all-MiniLM-L6-v2`) for simple semantic search
- Use **high-quality models** (e.g., `text-embedding-3-large`) for complex retrieval
- Consider **Matryoshka models** (e.g., `nomic-embed-text`) for flexible dimension requirements

### 2. Provider Configuration
- Configure embedding models at the **provider level** for consistency
- Use **environment variables** for API keys and sensitive configuration
- Set up **multiple providers** with different models for different use cases

### 3. Dimension Management
- Let the system **auto-lookup dimensions** when possible
- Only specify `embedding_dimension` for **Matryoshka embeddings** or custom requirements
- Ensure **model registry** has correct dimension metadata

### 4. Performance Optimization
- Use **smaller dimensions** for faster search (e.g., 256 instead of 768)
- Consider **multiple vector stores** with different embedding models for different content types
- Test **different embedding models** to find the best balance for your use case

## Troubleshooting

### Common Issues

**Model not found error:**
```
ValueError: Embedding model 'my-model' not found in model registry
```
**Solution:** Ensure the model is registered in your model configuration.

**Missing dimension metadata:**
```
ValueError: Embedding model 'my-model' has no embedding_dimension in metadata
```
**Solution:** Add `embedding_dimension` to the model's metadata in your model registry.

**Invalid dimension override:**
```
ValueError: Override dimension must be positive, got -1
```
**Solution:** Use positive integers for `embedding_dimension` values.

### Debugging Tips

1. **Check model registry:** Verify embedding models are properly registered
2. **Review provider config:** Ensure `embedding_model` matches registry IDs
3. **Test explicit parameters:** Override provider defaults to isolate issues
4. **Check logs:** Look for embedding model selection messages in router logs