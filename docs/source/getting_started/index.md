# Quickstart

Get started with Llama Stack in minutes!

In this guide, we'll walk through how to build a RAG application locally using Llama Stack with [Ollama](https://ollama.com/)
as the inference [provider](../providers/index.md#inference) for a Llama Model.

## Step 1. Install [uv](https://docs.astral.sh/uv/) and setup your virtual environment
```bash
uv pip install llama-stack aiosqlite faiss-cpu ollama \
openai datasets opentelemetry-exporter-otlp-proto-http mcp autoevals
source .venv/bin/activate
export INFERENCE_MODEL="llama3.2:3b"
```
## Step 2: Run inference locally with Ollama
```bash
# make sure to run this in a separate terminal
ollama run llama3.2:3b --keepalive 60m
```
## Step 3: Run the Demo
You can run this with `uv run demo_script.py` or in an interactive shell.
```python
from termcolor import cprint
from llama_stack.distribution.library_client import LlamaStackAsLibraryClient
from llama_stack_client.types import Document

vector_db = "faiss"
vector_db_id = "test-vector-db"
model_id = "llama3.2:3b-instruct-fp16"
query = "Can you give me the arxiv link for Lora Fine Tuning in Pytorch?"
documents = [
    Document(
        document_id="document_1",
        content=f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/lora_finetune.rst",
        mime_type="text/plain",
        metadata={},
    )
]

client = LlamaStackAsLibraryClient("ollama")
_ = client.initialize()
client.vector_dbs.register(
    provider_id=vector_db,
    vector_db_id=vector_db_id,
    embedding_model="all-MiniLM-L6-v2",
    embedding_dimension=384,
)

client.tool_runtime.rag_tool.insert(
    documents=documents,
    vector_db_id=vector_db_id,
    chunk_size_in_tokens=50,
)

response = client.tool_runtime.rag_tool.query(
    vector_db_ids=[vector_db_id],
    content=query,
)

cprint("" + "-" * 50, "yellow")
cprint(f"Query> {query}", "red")
cprint("" + "-" * 50, "yellow")
for chunk in response.content:
    cprint(f"Chunk ID> {chunk.text}", "green")
    cprint("" + "-" * 50, "yellow")
```
And you should see output like below.
```bash
--------------------------------------------------
Query> Can you give me the arxiv link for Lora Fine Tuning in Pytorch?
--------------------------------------------------
Chunk ID> knowledge_search tool found 5 chunks:
BEGIN of knowledge_search tool results.

--------------------------------------------------
Chunk ID> Result 1:
Document_id:docum
Content: .. _lora_finetune_label:

============================
Fine-Tuning Llama2 with LoRA
============================

This guide will teach you about `LoRA <https://arxiv.org/abs/2106.09685>`_, a

--------------------------------------------------
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
