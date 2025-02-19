---
orphan: true
---
# MongoDB Atlas

[MongoDB Atlas](https://www.mongodb.com/atlas) is a cloud database service that can be used as a vector store provider for Llama Stack. It supports vector search capabilities through its Atlas Vector Search feature, allowing you to store and query vectors within your MongoDB database.

## Features
MongoDB Atlas Vector Search supports:
- Store embeddings and their metadata
- Vector search with multiple algorithms (cosine similarity, euclidean distance, dot product)
- Hybrid search (combining vector and keyword search)
- Metadata filtering
- Scalable vector indexing
- Managed cloud infrastructure

## Usage

To use MongoDB Atlas in your Llama Stack project, follow these steps:

1. Create a MongoDB Atlas account and cluster.
2. Configure your Atlas cluster to enable Vector Search.
3. Configure your Llama Stack project to use MongoDB Atlas.
4. Start storing and querying vectors.

## Installation

You can install the MongoDB Python driver using pip:

```bash
pip install pymongo
```

## Documentation
See [MongoDB Atlas Vector Search documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/) for more details about vector search capabilities in MongoDB Atlas.
