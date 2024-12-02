# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import streamlit as st
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.memory_insert_params import Document

from modules.api import llama_stack_api
from modules.utils import data_url_from_file


def rag_chat_page():
    st.title("🦙 RAG")

    with st.sidebar:
        # File/Directory Upload Section
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload file(s) or directory",
            accept_multiple_files=True,
            type=["txt", "pdf", "doc", "docx"],  # Add more file types as needed
        )
        # Process uploaded files
        if uploaded_files:
            st.success(f"Successfully uploaded {len(uploaded_files)} files")
            # Add memory bank name input field
            memory_bank_name = st.text_input(
                "Memory Bank Name",
                value="rag_bank",
                help="Enter a unique identifier for this memory bank",
            )
            if st.button("Create Memory Bank"):
                documents = [
                    Document(
                        document_id=uploaded_file.name,
                        content=data_url_from_file(uploaded_file),
                    )
                    for i, uploaded_file in enumerate(uploaded_files)
                ]

                providers = llama_stack_api.client.providers.list()
                llama_stack_api.client.memory_banks.register(
                    memory_bank_id=memory_bank_name,  # Use the user-provided name
                    params={
                        "embedding_model": "all-MiniLM-L6-v2",
                        "chunk_size_in_tokens": 512,
                        "overlap_size_in_tokens": 64,
                    },
                    provider_id=providers["memory"][0].provider_id,
                )

                # insert documents using the custom bank name
                llama_stack_api.client.memory.insert(
                    bank_id=memory_bank_name,  # Use the user-provided name
                    documents=documents,
                )
                st.success("Memory bank created successfully!")

        st.subheader("Configure Agent")
        # select memory banks
        memory_banks = llama_stack_api.client.memory_banks.list()
        memory_banks = [bank.identifier for bank in memory_banks]
        selected_memory_banks = st.multiselect(
            "Select Memory Banks",
            memory_banks,
        )
        memory_bank_configs = [
            {"bank_id": bank_id, "type": "vector"} for bank_id in selected_memory_banks
        ]

        available_models = llama_stack_api.client.models.list()
        available_models = [model.identifier for model in available_models]
        selected_model = st.selectbox(
            "Choose a model",
            available_models,
            index=0,
        )
        system_prompt = st.text_area(
            "System Prompt",
            value="You are a helpful assistant. ",
            help="Initial instructions given to the AI to set its behavior and context",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Controls the randomness of the response. Higher values make the output more creative and unexpected, lower values make it more conservative and predictable",
        )

        top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.1,
        )

        # Add clear chat button to sidebar
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    selected_model = llama_stack_api.client.models.list()[0].identifier

    agent_config = AgentConfig(
        model=selected_model,
        instructions=system_prompt,
        sampling_params={
            "strategy": "greedy",
            "temperature": temperature,
            "top_p": top_p,
        },
        tools=[
            {
                "type": "memory",
                "memory_bank_configs": memory_bank_configs,
                "query_generator_config": {"type": "default", "sep": " "},
                "max_tokens_in_context": 4096,
                "max_chunks": 10,
            }
        ],
        tool_choice="auto",
        tool_prompt_format="json",
        input_shields=[],
        output_shields=[],
        enable_session_persistence=False,
    )

    agent = Agent(llama_stack_api.client, agent_config)
    session_id = agent.create_session("rag-session")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        response = agent.create_turn(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            session_id=session_id,
        )

        # Display assistant response
        with st.chat_message("assistant"):
            retrieval_message_placeholder = st.empty()
            message_placeholder = st.empty()
            full_response = ""
            retrieval_response = ""
            for log in EventLogger().log(response):
                log.print()
                if log.role == "memory_retrieval":
                    retrieval_response += log.content.replace("====", "").strip()
                    retrieval_message_placeholder.info(retrieval_response)
                else:
                    full_response += log.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )


rag_chat_page()
