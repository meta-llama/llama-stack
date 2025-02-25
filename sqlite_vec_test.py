#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import uuid

from termcolor import cprint

# Set environment variables
os.environ["INFERENCE_MODEL"] = "llama3.2:3b-instruct-fp16"
os.environ["LLAMA_STACK_CONFIG"] = "ollama"

# Import libraries after setting environment variables
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types import Document
from llama_stack_client.types.agent_create_params import AgentConfig

from llama_stack.distribution.library_client import LlamaStackAsLibraryClient


def main():
    # Initialize the client
    client = LlamaStackAsLibraryClient("ollama")
    vector_db_id = f"test-vector-db-{uuid.uuid4().hex}"

    _ = client.initialize()

    model_id = "llama3.2:3b-instruct-fp16"

    # Define the list of document URLs and create Document objects
    urls = [
        "chat.rst",
        "llama3.rst",
        "memory_optimizations.rst",
        "lora_finetune.rst",
    ]
    documents = [
        Document(
            document_id=f"num-{i}",
            content=f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}",
            mime_type="text/plain",
            metadata={},
        )
        for i, url in enumerate(urls)
    ]
    # (Optional) Use the documents as needed with your client here

    client.vector_dbs.register(
        provider_id="sqlite_vec",
        vector_db_id=vector_db_id,
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
    )

    client.tool_runtime.rag_tool.insert(
        documents=documents,
        vector_db_id=vector_db_id,
        chunk_size_in_tokens=512,
    )
    # Create agent configuration
    agent_config = AgentConfig(
        model=model_id,
        instructions="You are a helpful assistant",
        enable_session_persistence=False,
        toolgroups=[
            {
                "name": "builtin::rag",
                "args": {
                    "vector_db_ids": [vector_db_id],
                },
            }
        ],
    )

    # Instantiate the Agent
    agent = Agent(client, agent_config)

    # List of user prompts
    user_prompts = [
        "What are the top 5 topics that were explained in the documentation? Only list succinct bullet points.",
        "Was anything related to 'Llama3' discussed, if so what?",
        "Tell me how to use LoRA",
        "What about Quantization?",
    ]

    # Create a session for the agent
    session_id = agent.create_session("test-session")

    # Process each prompt and display the output
    for prompt in user_prompts:
        cprint(f"User> {prompt}", "green")
        response = agent.create_turn(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            session_id=session_id,
        )
        # Log and print events from the response
        for log in EventLogger().log(response):
            log.print()


if __name__ == "__main__":
    main()
