# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid

import streamlit as st
from llama_stack_client import Agent

from llama_stack.distribution.ui.modules.api import llama_stack_api


def tool_chat_page():
    st.title("🛠 Tools")

    client = llama_stack_api.client
    models = client.models.list()
    model_list = [model.identifier for model in models if model.api_model_type == "llm"]

    tool_groups = client.toolgroups.list()
    tool_groups_list = [tool_group.identifier for tool_group in tool_groups]
    mcp_tools_list = [tool for tool in tool_groups_list if tool.startswith("mcp::")]
    builtin_tools_list = [tool for tool in tool_groups_list if not tool.startswith("mcp::")]

    def reset_agent():
        st.session_state.clear()
        st.cache_resource.clear()

    with st.sidebar:
        st.subheader("Model")
        model = st.selectbox(label="models", options=model_list, on_change=reset_agent)

        st.subheader("Builtin Tools")
        toolgroup_selection = st.pills(
            label="Available ToolGroups", options=builtin_tools_list, selection_mode="multi", on_change=reset_agent
        )

        st.subheader("MCP Servers")
        mcp_selection = st.pills(
            label="Available MCP Servers", options=mcp_tools_list, selection_mode="multi", on_change=reset_agent
        )

        toolgroup_selection.extend(mcp_selection)

        active_tool_list = []
        for toolgroup_id in toolgroup_selection:
            active_tool_list.extend(
                [
                    f"{''.join(toolgroup_id.split('::')[1:])}:{t.identifier}"
                    for t in client.tools.list(toolgroup_id=toolgroup_id)
                ]
            )

        st.subheader(f"Active Tools: 🛠 {len(active_tool_list)}")
        st.json(active_tool_list)

    @st.cache_resource
    def create_agent():
        return Agent(
            client,
            model=model,
            instructions="You are a helpful assistant. When you use a tool always respond with a summary of the result.",
            tools=toolgroup_selection,
            sampling_params={
                "strategy": {"type": "greedy"},
            },
        )

    agent = create_agent()

    if "agent_session_id" not in st.session_state:
        st.session_state["agent_session_id"] = agent.create_session(session_name=f"tool_demo_{uuid.uuid4()}")

    session_id = st.session_state["agent_session_id"]

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input(placeholder=""):
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        turn_response = agent.create_turn(
            session_id=session_id,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        def response_generator(turn_response):
            for response in turn_response:
                if hasattr(response.event, "payload"):
                    print(response.event.payload)
                    if response.event.payload.event_type == "step_progress":
                        if hasattr(response.event.payload.delta, "text"):
                            yield response.event.payload.delta.text
                    if response.event.payload.event_type == "step_complete":
                        if response.event.payload.step_details.step_type == "tool_execution":
                            yield " 🛠 "
                else:
                    yield f"Error occurred in the Llama Stack Cluster: {response}"

        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(turn_response))

        st.session_state.messages.append({"role": "assistant", "content": response})


tool_chat_page()
