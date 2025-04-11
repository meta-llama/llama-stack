# Quickstart

Get started with Llama Stack in minutes!

Llama Stack is a stateful service with REST APIs to support the seamless transition of AI applications across different
environments. You can build and test using a local server first and deploy to a hosted endpoint for production.

In this guide, we'll walk through how to build a RAG application locally using Llama Stack with [Ollama](https://ollama.com/)
as the inference [provider](../providers/index.md#inference) for a Llama Model.

## Step 1. Install and Setup
Install [uv](https://docs.astral.sh/uv/), setup your virtual environment, and run inference on a Llama model with
[Ollama](https://ollama.com/download).
```bash
uv pip install llama-stack
source .venv/bin/activate
ollama run llama3.2:3b --keepalive 60m
```
## Step 2: Run the Llama Stack Server
```bash
INFERENCE_MODEL=llama3.2:3b llama stack build --template ollama --image-type venv --run
```
## Step 3: Run the Demo
Now open up a new terminal using the same virtual environment and you can run this demo as a script using `uv run demo_script.py` or in an interactive shell.
```python
from llama_stack_client import Agent, AgentEventLogger, RAGDocument, LlamaStackClient

vector_db_id = "my_demo_vector_db"
client = LlamaStackClient(base_url="http://localhost:8321")

models = client.models.list()

# Select the first LLM and first embedding models
model_id = next(m for m in models if m.model_type == "llm").identifier
embedding_model_id = (
    em := next(m for m in models if m.model_type == "embedding")
).identifier
embedding_dimension = em.metadata["embedding_dimension"]

_ = client.vector_dbs.register(
    vector_db_id=vector_db_id,
    embedding_model=embedding_model_id,
    embedding_dimension=embedding_dimension,
    provider_id="faiss",
)
document = RAGDocument(
    document_id="document_1",
    content="https://www.paulgraham.com/greatwork.html",
    mime_type="text/html",
    metadata={},
)
client.tool_runtime.rag_tool.insert(
    documents=[document],
    vector_db_id=vector_db_id,
    chunk_size_in_tokens=50,
)
agent = Agent(
    client,
    model=model_id,
    instructions="You are a helpful assistant",
    tools=[
        {
            "name": "builtin::rag/knowledge_search",
            "args": {"vector_db_ids": [vector_db_id]},
        }
    ],
)

response = agent.create_turn(
    messages=[{"role": "user", "content": "How do you do great work?"}],
    session_id=agent.create_session("rag_session"),
)

for log in AgentEventLogger().log(response):
    log.print()
```
And you should see output like below.
```bash
inference> [knowledge_search(query="What does it mean to do great work")]
tool_execution> Tool:knowledge_search Args:{'query': 'What does it mean to do great work'}
tool_execution> Tool:knowledge_search Response:[TextContentItem(text='knowledge_search tool found 5 chunks:\nBEGIN of knowledge_search tool results.\n', type='text'), TextContentItem(text="Result 1:\nDocument_id:docum\nContent:  work. Doing great work means doing something important\nso well that you expand people's ideas of what's possible. But\nthere's no threshold for importance. It's a matter of degree, and\noften hard to judge at the time anyway.\n", type='text'), TextContentItem(text='Result 2:\nDocument_id:docum\nContent: [<a name="f1n"><font color=#000000>1</font></a>]\nI don\'t think you could give a precise definition of what\ncounts as great work. Doing great work means doing something important\nso well\n', type='text'), TextContentItem(text="Result 3:\nDocument_id:docum\nContent: . And if so\nyou're already further along than you might realize, because the\nset of people willing to want to is small.<br /><br />The factors in doing great work are factors in the literal,\nmathematical sense, and\n", type='text'), TextContentItem(text="Result 4:\nDocument_id:docum\nContent: \nincreases your morale and helps you do even better work. But this\ncycle also operates in the other direction: if you're not doing\ngood work, that can demoralize you and make it even harder to. Since\nit matters\n", type='text'), TextContentItem(text="Result 5:\nDocument_id:docum\nContent:  to try to do\ngreat work. But that's what's going on subconsciously; they shy\naway from the question.<br /><br />So I'm going to pull a sneaky trick on you. Do you want to do great\n", type='text'), TextContentItem(text='END of knowledge_search tool results.\n', type='text')]
```
Congratulations! You've successfully built your first RAG application using Llama Stack! 🎉🥳

## Next Steps

Now you're ready to dive deeper into Llama Stack!
- Explore the [Detailed Tutorial](./detailed_tutorial.md).
- Try the [Getting Started Notebook](https://github.com/meta-llama/llama-stack/blob/main/docs/getting_started.ipynb).
- Browse more [Notebooks on GitHub](https://github.com/meta-llama/llama-stack/tree/main/docs/notebooks).
- Learn about Llama Stack [Concepts](../concepts/index.md).
- Discover how to [Build Llama Stacks](../distributions/index.md).
- Refer to our [References](../references/index.md) for details on the Llama CLI and Python SDK.
- Check out the [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repository for example applications and tutorials.

```{toctree}
:maxdepth: 0
:hidden:

detailed_tutorial
```
