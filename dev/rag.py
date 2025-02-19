import os
import uuid
from termcolor import cprint

from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types import Document


def create_http_client():
    from llama_stack_client import LlamaStackClient

    return LlamaStackClient(
        base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}"
    )


def create_library_client(template="ollama"):
    from llama_stack import LlamaStackAsLibraryClient

    client = LlamaStackAsLibraryClient(template)
    client.initialize()
    return client


client = (
    create_library_client()
)  # or create_http_client() depending on the environment you picked

# Documents to be used for RAG
urls = ["chat.rst", "llama3.rst", "memory_optimizations.rst", "lora_finetune.rst"]
documents = [
    Document(
        document_id=f"num-{i}",
        content=f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}",
        mime_type="text/plain",
        metadata={},
    )
    for i, url in enumerate(urls)
]

# Register a vector database
vector_db_id = f"test-vector-db-{uuid.uuid4().hex}"
client.vector_dbs.register(
    vector_db_id=vector_db_id,
    embedding_model="all-MiniLM-L6-v2",
    embedding_dimension=384,
    provider_id="sqlite_vec",
)

# Insert the documents into the vector database
client.tool_runtime.rag_tool.insert(
    documents=documents,
    vector_db_id=vector_db_id,
    chunk_size_in_tokens=512,
)

agent_config = AgentConfig(
    model=os.environ["INFERENCE_MODEL"],
    # Define instructions for the agent ( aka system prompt)
    instructions="You are a helpful assistant",
    enable_session_persistence=False,
    # Define tools available to the agent
    toolgroups=[
        {
            "name": "builtin::rag",
            "args": {
                "vector_db_ids": [vector_db_id],
            },
        }
    ],
)

rag_agent = Agent(client, agent_config)
session_id = rag_agent.create_session("test-session")

user_prompts = [
    "What are the top 5 topics that were explained? Only list succinct bullet points.",
]

# Run the agent loop by calling the `create_turn` method
for prompt in user_prompts:
    cprint(f"User> {prompt}", "green")
    response = rag_agent.create_turn(
        messages=[{"role": "user", "content": prompt}],
        session_id=session_id,
    )
    for log in EventLogger().log(response):
        log.print()