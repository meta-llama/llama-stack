# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os
import streamlit as st
from typing import Dict, List, Optional, Any

from llama_stack.distribution.ui.modules.api import llama_stack_api

# Constants
CONFIG_DIR = os.path.expanduser("~/.llama_stack/mcp_servers")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

# Ensure config directory exists
os.makedirs(CONFIG_DIR, exist_ok=True)

def load_config() -> Dict[str, Any]:
    """Load MCP server configurations from file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.error("Error loading MCP server configuration file. Using default configuration.")
    
    # Default empty configuration
    return {
        "servers": {},
        "inputs": []
    }

def register_mcp_servers(config: Dict[str, Any]) -> None:
    """Register MCP servers as toolgroups with the LlamaStackClient."""
    servers = config.get("servers", {})
    inputs = config.get("inputs", [])
    
    # Process inputs to resolve values
    input_values = {}
    for input_config in inputs:
        input_id = input_config.get("id")
        if input_id:
            # Get value from session state if available
            input_values[input_id] = st.session_state.get(f"input_value_{input_id}", "")
    
    # Update provider data with MCP headers
    mcp_headers = {}
    
    # Register each server as a toolgroup
    for server_name, server_config in servers.items():
        url = server_config.get("url", "")
        if not url:
            continue
        
        try:
            # Register the MCP server as a toolgroup
            toolgroup_id = f"mcp::{server_name}"
            
            # Register the toolgroup
            llama_stack_api.client.toolgroups.register(
                toolgroup_id=toolgroup_id,
                provider_id="model-context-protocol",
                mcp_endpoint=url,
            )
            
            # Process headers for this server
            headers = server_config.get("headers", {})
            processed_headers = {}
            
            for header_key, header_value in headers.items():
                # Process input references in header values
                if isinstance(header_value, str) and "${input:" in header_value:
                    # Extract input ID from ${input:input_id} format
                    input_id = header_value.split("${input:")[1].split("}")[0]
                    if input_id in input_values:
                        processed_headers[header_key] = input_values[input_id]
                else:
                    processed_headers[header_key] = header_value
            
            # Add headers to mcp_headers if there are any
            if processed_headers:
                mcp_headers[url] = processed_headers
                
        except Exception as e:
            st.error(f"Failed to register MCP server '{server_name}': {str(e)}")
    
    # Update provider data with MCP headers if there are any
    if mcp_headers:
        provider_data = llama_stack_api.provider_data.copy()
        provider_data["mcp_headers"] = mcp_headers
        llama_stack_api.update_provider_data_dict(provider_data)

def save_config(config: Dict[str, Any]) -> None:
    """Save MCP server configurations to file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    
    # Register MCP servers as toolgroups
    register_mcp_servers(config)
    
    st.success("MCP server configuration saved successfully!")

def render_server_config(server_name: str, server_config: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Render and edit configuration for a specific MCP server."""
    st.subheader(f"Server: {server_name}")
    
    # Server type
    server_type = st.selectbox(
        "Server Type", 
        ["http", "websocket"], 
        index=0 if server_config.get("type", "http") == "http" else 1,
        key=f"type_{server_name}"
    )
    
    # Server URL
    url = st.text_input(
        "Server URL", 
        value=server_config.get("url", ""), 
        key=f"url_{server_name}"
    )
    
    # Headers
    st.write("Headers:")
    headers = server_config.get("headers", {})
    headers_container = st.container()
    
    with headers_container:
        # Display existing headers
        for header_key, header_value in list(headers.items()):
            col1, col2, col3 = st.columns([3, 6, 1])
            with col1:
                st.text(header_key)
            with col2:
                # Check if this is a reference to an input
                if isinstance(header_value, str) and "${input:" in header_value:
                    st.text(header_value)
                else:
                    st.text("********" if "token" in header_key.lower() or "auth" in header_key.lower() else header_value)
            with col3:
                if st.button("🗑️", key=f"delete_header_{server_name}_{header_key}"):
                    del headers[header_key]
    
    # Add new header
    st.write("Add Header:")
    header_col1, header_col2 = st.columns([1, 1])
    new_header_key = header_col1.text_input("Key", key=f"new_header_key_{server_name}")
    new_header_value = header_col2.text_input("Value", key=f"new_header_value_{server_name}")
    
    if st.button("Add Header", key=f"add_header_{server_name}"):
        if new_header_key and new_header_value:
            headers[new_header_key] = new_header_value
            st.experimental_rerun()
    
    # Construct updated server config
    updated_server_config = {
        "type": server_type,
        "url": url,
        "headers": headers
    }
    
    return updated_server_config

def render_input_config(input_config: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Render and edit configuration for an input field."""
    st.subheader(f"Input: {input_config.get('id', '')}")
    
    col1, col2 = st.columns([1, 1])
    
    input_type = col1.selectbox(
        "Type", 
        ["promptString"], 
        index=0,
        key=f"input_type_{index}"
    )
    
    input_id = col2.text_input(
        "ID", 
        value=input_config.get("id", ""),
        key=f"input_id_{index}"
    )
    
    description = st.text_input(
        "Description", 
        value=input_config.get("description", ""),
        key=f"input_desc_{index}"
    )
    
    is_password = st.checkbox(
        "Password Field", 
        value=input_config.get("password", False),
        key=f"input_password_{index}"
    )
    
    # Construct updated input config
    updated_input_config = {
        "type": input_type,
        "id": input_id,
        "description": description,
        "password": is_password
    }
    
    return updated_input_config

def main():
    st.title("MCP Servers Configuration")
    
    st.markdown("""
    Configure Model Control Protocol (MCP) servers to integrate with external AI services.
    MCP is a protocol for controlling AI models through a standardized API.
    
    MCP servers are registered as toolgroups with the ID format `mcp::{server_name}`.
    """)
    
    # Load existing configuration
    config = load_config()
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Servers", "Inputs", "Input Values", "JSON Config"])
    
    # Servers Tab
    with tab1:
        st.header("Configured Servers")
        
        # List existing servers
        servers = config.get("servers", {})
        if not servers:
            st.info("No MCP servers configured yet. Add a new server below.")
        
        # Server selection or creation
        server_options = list(servers.keys()) + ["+ Add New Server"]
        selected_server = st.selectbox("Select Server", server_options)
        
        if selected_server == "+ Add New Server":
            new_server_name = st.text_input("New Server Name")
            if new_server_name and st.button("Create Server"):
                if new_server_name in servers:
                    st.error(f"Server '{new_server_name}' already exists.")
                else:
                    servers[new_server_name] = {
                        "type": "http",
                        "url": "",
                        "headers": {}
                    }
                    st.experimental_rerun()
        elif selected_server in servers:
            # Edit existing server
            updated_config = render_server_config(selected_server, servers[selected_server], config)
            
            col1, col2 = st.columns([1, 1])
            if col1.button("Update Server", key=f"update_{selected_server}"):
                servers[selected_server] = updated_config
                save_config(config)
            
            if col2.button("Delete Server", key=f"delete_{selected_server}"):
                del servers[selected_server]
                save_config(config)
                st.experimental_rerun()
    
    # Inputs Tab
    with tab2:
        st.header("Input Configurations")
        
        inputs = config.get("inputs", [])
        if not inputs:
            st.info("No input configurations defined yet. Add a new input below.")
        
        # Input selection or creation
        input_options = [f"{i.get('id', f'Input {idx}')} ({i.get('type', 'promptString')})" for idx, i in enumerate(inputs)]
        input_options.append("+ Add New Input")
        
        selected_input_option = st.selectbox("Select Input", input_options)
        
        if selected_input_option == "+ Add New Input":
            if st.button("Create Input"):
                inputs.append({
                    "type": "promptString",
                    "id": f"input_{len(inputs)}",
                    "description": "",
                    "password": False
                })
                save_config(config)
                st.experimental_rerun()
        else:
            # Edit existing input
            selected_idx = input_options.index(selected_input_option)
            if selected_idx < len(inputs):
                updated_input = render_input_config(inputs[selected_idx], selected_idx)
                
                col1, col2 = st.columns([1, 1])
                if col1.button("Update Input", key=f"update_input_{selected_idx}"):
                    inputs[selected_idx] = updated_input
                    save_config(config)
                
                if col2.button("Delete Input", key=f"delete_input_{selected_idx}"):
                    inputs.pop(selected_idx)
                    save_config(config)
                    st.experimental_rerun()
    
    # Input Values Tab
    with tab3:
        st.header("Input Values")
        
        inputs = config.get("inputs", [])
        if not inputs:
            st.info("No input configurations defined yet. Add inputs in the Inputs tab.")
        else:
            st.write("Enter values for the configured inputs:")
            
            for input_config in inputs:
                input_id = input_config.get("id", "")
                description = input_config.get("description", "")
                is_password = input_config.get("password", False)
                
                # Get value from session state if available
                current_value = st.session_state.get(f"input_value_{input_id}", "")
                
                # Input field for value
                value = st.text_input(
                    f"{description} ({input_id})",
                    value=current_value,
                    type="password" if is_password else "default",
                    key=f"input_value_{input_id}"
                )
            
            if st.button("Save Input Values"):
                # Values are automatically saved to session state
                # Register MCP servers to apply the new values
                register_mcp_servers(config)
                st.success("Input values saved successfully!")
    
    # JSON Config Tab
    with tab4:
        st.header("Raw JSON Configuration")
        
        # Display and edit raw JSON
        json_str = json.dumps(config, indent=2)
        edited_json = st.text_area("Edit JSON Configuration", json_str, height=400)
        
        if st.button("Update from JSON"):
            try:
                updated_config = json.loads(edited_json)
                save_config(updated_config)
                st.experimental_rerun()
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {str(e)}")
    
    # Example configuration and usage
    with st.expander("Example Configuration and Usage"):
        st.markdown("""
        ### Example Configuration
        
        ```json
        {
          "servers": {
            "github": {
              "type": "http",
              "url": "https://api.githubcopilot.com/mcp/",
              "headers": {
                "Authorization": "Bearer ${input:github_mcp_pat}"
              }
            }
          },
          "inputs": [
            {
              "type": "promptString",
              "id": "github_mcp_pat",
              "description": "GitHub Personal Access Token",
              "password": true
            }
          ]
        }
        ```
        
        ### Usage in Code
        
        Once registered, you can use the MCP server in your code:
        
        ```python
        from llama_stack_client import Agent
        
        agent = Agent(
            model="llama-3-70b-instruct",
            tools=["mcp::github"],  # Use the registered MCP server
        )
        
        agent.create_turn("Use GitHub Copilot to help me write a function")
        ```
        """)

if __name__ == "__main__":
    main()
