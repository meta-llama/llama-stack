# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import streamlit as st
from llama_stack_client import Agent, AgentEventLogger, RAGDocument

from llama_stack.distribution.ui.modules.api import llama_stack_api
from llama_stack.distribution.ui.modules.utils import data_url_from_file


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
            vector_db_name = st.text_input(
                "Vector Database Name",
                value="rag_vector_db",
                help="Enter a unique identifier for this vector database",
            )
            if st.button("Create Vector Database"):
                documents = [
                    RAGDocument(
                        document_id=uploaded_file.name,
                        content=data_url_from_file(uploaded_file),
                    )
                    for i, uploaded_file in enumerate(uploaded_files)
                ]

                providers = llama_stack_api.client.providers.list()
                vector_io_provider = None

                for x in providers:
                    if x.api == "vector_io":
                        vector_io_provider = x.provider_id

                llama_stack_api.client.vector_dbs.register(
                    vector_db_id=vector_db_name,  # Use the user-provided name
                    embedding_dimension=384,
                    embedding_model="all-MiniLM-L6-v2",
                    provider_id=vector_io_provider,
                )

                # insert documents using the custom vector db name
                llama_stack_api.client.tool_runtime.rag_tool.insert(
                    vector_db_id=vector_db_name,  # Use the user-provided name
                    documents=documents,
                )
                st.success("Vector database created successfully!")

        st.subheader("Configure Agent")
        # select memory banks
        vector_dbs = llama_stack_api.client.vector_dbs.list()
        vector_dbs = [vector_db.identifier for vector_db in vector_dbs]
        selected_vector_dbs = st.multiselect(
            "Select Vector Databases",
            vector_dbs,
        )

        available_models = llama_stack_api.client.models.list()
        available_models = [model.identifier for model in available_models if model.model_type == "llm"]
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

    if temperature > 0.0:
        strategy = {
            "type": "top_p",
            "temperature": temperature,
            "top_p": top_p,
        }
    else:
        strategy = {"type": "greedy"}

    agent = Agent(
        llama_stack_api.client,
        model=selected_model,
        instructions=system_prompt,
        sampling_params={
            "strategy": strategy,
        },
        tools=[
            dict(
                name="builtin::rag/knowledge_search",
                args={
                    "vector_db_ids": list(selected_vector_dbs),
                },
            )
        ],
    )
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
            for log in AgentEventLogger().log(response):
                log.print()
                if log.role == "tool_execution":
                    retrieval_response += log.content.replace("====", "").strip()
                    retrieval_message_placeholder.info(retrieval_response)
                else:
                    full_response += log.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})


rag_chat_page()
