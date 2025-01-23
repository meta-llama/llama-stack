## Memory & RAG

Memory enables your applications to reference and recall information from previous interactions or external documents. Llama Stack's memory system is built around the concept of Memory Banks:

1. **Vector Memory Banks**: For semantic search and retrieval
2. **Key-Value Memory Banks**: For structured data storage
3. **Keyword Memory Banks**: For basic text search
4. **Graph Memory Banks**: For relationship-based retrieval

Here's how to set up a vector memory bank for RAG:

```python
# Register a memory bank
bank_id = "my_documents"
response = client.memory_banks.register(
    memory_bank_id=bank_id,
    params={
        "memory_bank_type": "vector",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size_in_tokens": 512
    }
)

# Insert documents
documents = [
    {
        "document_id": "doc1",
        "content": "Your document text here",
        "mime_type": "text/plain"
    }
]
client.memory.insert(bank_id, documents)

# Query documents
results = client.memory.query(
    bank_id=bank_id,
    query="What do you know about...",
)
```


### Building RAG-Enhanced Agents

One of the most powerful patterns is combining agents with RAG capabilities. Here's a complete example:

```python
from llama_stack_client.types import Attachment

# Create attachments from documents
attachments = [
    Attachment(
        content="https://raw.githubusercontent.com/example/doc.rst",
        mime_type="text/plain"
    )
]

# Configure agent with memory
agent_config = AgentConfig(
    model="Llama3.2-3B-Instruct",
    instructions="You are a helpful assistant",
    tools=[{
        "type": "memory",
        "memory_bank_configs": [],
        "query_generator_config": {"type": "default", "sep": " "},
        "max_tokens_in_context": 4096,
        "max_chunks": 10
    }],
    enable_session_persistence=True
)

agent = Agent(client, agent_config)
session_id = agent.create_session("rag_session")

# Initial document ingestion
response = agent.create_turn(
    messages=[{
        "role": "user",
        "content": "I am providing some documents for reference."
    }],
    attachments=attachments,
    session_id=session_id
)

# Query with RAG
response = agent.create_turn(
    messages=[{
        "role": "user",
        "content": "What are the key topics in the documents?"
    }],
    session_id=session_id
)
```
