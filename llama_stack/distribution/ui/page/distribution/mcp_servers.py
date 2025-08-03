# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os
import streamlit as st

from llama_stack.distribution.ui.modules.api import llama_stack_api
from llama_stack_client import LlamaStackClient
from llama_stack_client.types.toolgroup_register_params import McpEndpoint


def main():
    st.title("MCP Servers Configuration")
    
    st.markdown("""
    Configure Model Control Protocol (MCP) servers to integrate with external AI services.
    MCP is a protocol for controlling AI models through a standardized API.
    
    MCP servers are registered as toolgroups with the ID format `mcp::{server_name}`.
    """)
    
    # MCP Server Configuration
    st.header("MCP Server Configuration")
    
    # Create a form for MCP configuration
    with st.form("mcp_server_form"):
        # Server name
        server_name = st.text_input(
            "Server Name",
            value="github",
            help="A unique name for this MCP server. Will be used in the toolgroup ID as mcp::{name}."
        )
        
        # MCP URL
        mcp_url = st.text_input(
            "MCP URL",
            value="https://api.githubcopilot.com/mcp/",
            help="The URL of the MCP server."
        )
        
        # Get the current value from session state
        api_token = st.session_state.get(f"mcp_token_{server_name}", "")
        
        # Input field for API Bearer Token
        mcp_token = st.text_input(
            "API Bearer Token", 
            value=api_token,
            type="password",
            help="Enter your API Bearer Token. For GitHub Copilot, this should be a GitHub Personal Access Token with Copilot scope."
        )
        
        
        # Submit button
        submit_button = st.form_submit_button("Save Configuration")
        
        if submit_button:
            if not server_name:
                st.error("Server name is required.")
            elif not mcp_url:
                st.error("MCP URL is required.")
            else:
                # Store the token in session state
                st.session_state[f"mcp_token_{server_name}"] = mcp_token
                
                try:
                    # Register the MCP server as a toolgroup
                    toolgroup_id = f"mcp::{server_name}"
                    try:
                        llama_stack_api.client.toolgroups.register(
                            toolgroup_id=toolgroup_id,
                            provider_id="model-context-protocol",
                            mcp_endpoint=McpEndpoint(uri=mcp_url),
                            timeout=10.0,  # Set a reasonable timeout
                        )
                    except Exception as e:
                        if "timeout" in str(e).lower():
                            st.warning(f"Registration timed out, but configuration will still be saved. Error: {str(e)}")
                        else:
                            raise
                    
                    # Update provider data with MCP headers
                    # Check if provider_data attribute exists
                    if not hasattr(llama_stack_api, "provider_data"):
                        llama_stack_api.provider_data = {}
                    
                    provider_data = llama_stack_api.provider_data.copy()
                    if "mcp_headers" not in provider_data:
                        provider_data["mcp_headers"] = {}
                    
                    # Add MCP headers
                    if mcp_token:
                        # Clean the token (remove 'Bearer ' prefix if present)
                        clean_token = mcp_token
                        if clean_token.lower().startswith("bearer "):
                            clean_token = clean_token[7:]
                        
                        # Set the headers for this MCP server
                        # The key needs to be the exact URL used in the MCP endpoint
                        provider_data["mcp_headers"][mcp_url] = {
                            "Authorization": f"Bearer {clean_token}"
                        }
                        
                        # Debug information
                        st.info(f"Set authentication headers for {mcp_url}: Bearer {clean_token[:4]}...")
                        
                        # Also set the token directly in the provider_data using different formats
                        # This increases the chance that one of them will work
                        provider_data[f"{server_name}_api_key"] = clean_token
                        provider_data[f"{server_name}_token"] = clean_token
                        provider_data[f"{server_name}_mcp_token"] = clean_token
                        
                        # Display the current provider_data for debugging
                        st.write("Current provider_data:")
                        
                        # Mask the token for display
                        masked_token = ""
                        if clean_token:
                            if len(clean_token) > 8:
                                # Show first 2 and last 2 characters, mask the middle with asterisks
                                masked_token = f"Bearer {clean_token[:2]}{'*' * 6}{clean_token[-2:]}"
                            else:
                                # For short tokens, just show first 2 chars and asterisks
                                masked_token = f"Bearer {clean_token[:2]}{'*' * 4}"
                        
                        # Create a sanitized version of mcp_headers for display
                        sanitized_headers = {}
                        for url, headers in provider_data.get("mcp_headers", {}).items():
                            sanitized_headers[url] = {}
                            for header_key, header_value in headers.items():
                                if header_key.lower() == "authorization" and isinstance(header_value, str):
                                    if header_value.lower().startswith("bearer "):
                                        token = header_value[7:]
                                        if len(token) > 8:
                                            sanitized_headers[url][header_key] = f"Bearer {token[:2]}{'*' * 6}{token[-2:]}"
                                        else:
                                            sanitized_headers[url][header_key] = f"Bearer {token[:2]}{'*' * 4}"
                                    else:
                                        sanitized_headers[url][header_key] = f"{header_value[:2]}{'*' * 6}"
                                else:
                                    sanitized_headers[url][header_key] = header_value
                        
                        st.json({
                            "mcp_headers": sanitized_headers,
                            f"{server_name}_api_key": masked_token
                        })
                        
                        # Add a note about authentication
                        st.warning("""
                        **Important Authentication Notes:**
                        
                        1. If you encounter authentication errors when using this MCP server, 
                           try restarting the UI application to ensure the authentication headers are properly applied.
                        
                        2. For GitHub Copilot, make sure you're using a valid GitHub Personal Access Token with the 'copilot' scope.
                        
                        3. When using the MCP server in code, you may need to explicitly pass the authentication headers:
                           ```python
                           import json
                           from llama_stack_client import Agent
                           
                           agent = Agent(
                               model="llama-3-70b-instruct",
                               tools=["mcp::github"],
                               extra_headers={
                                   "X-LlamaStack-Provider-Data": json.dumps({
                                       "mcp_headers": {
                                           "https://api.githubcopilot.com/mcp/": {
                                               "Authorization": "Bearer YOUR_TOKEN_HERE"
                                           }
                                       }
                                   })
                               }
                           )
                           ```
                        """)
                    
                    # Update the client with the new provider data
                    if hasattr(llama_stack_api, "update_provider_data_dict"):
                        llama_stack_api.update_provider_data_dict(provider_data)
                    else:
                        # Fallback implementation if method doesn't exist
                        llama_stack_api.provider_data = provider_data
                        # Reinitialize the client with updated provider data
                        llama_stack_api.client = LlamaStackClient(
                            base_url=os.environ.get("LLAMA_STACK_ENDPOINT", "http://localhost:8321"),
                            provider_data=provider_data,
                        )
                    
                    st.success(f"MCP server '{server_name}' configured successfully!")
                except Exception as e:
                    st.error(f"Failed to configure MCP server '{server_name}': {str(e)}")
    
    # Usage example
    with st.expander("Usage Example"):
        st.markdown("""
        ### Using MCP Servers in Code
        
        Once configured, you can use the MCP server in your code:
        
        ```python
        from llama_stack_client import Agent
        
        agent = Agent(
            model="llama-3-70b-instruct",
            tools=["mcp::your_server_name"],  # Use the registered MCP server
        )
        
        agent.create_turn("Use the MCP server to help me with a task")
        ```
        
        ### With Explicit Authentication Headers
        
        If you encounter authentication issues, you can explicitly pass the authentication headers:
        
        ```python
        import json
        from llama_stack_client import Agent
        
        agent = Agent(
            model="llama-3-70b-instruct",
            tools=["mcp::github"],
            extra_headers={
                "X-LlamaStack-Provider-Data": json.dumps({
                    "mcp_headers": {
                        "https://api.githubcopilot.com/mcp/": {
                            "Authorization": "Bearer YOUR_TOKEN_HERE"
                        }
                    }
                })
            }
        )
        ```
        """)
    
    # Display registered toolgroups
    st.header("Registered MCP Toolgroups")
    try:
        toolgroups = llama_stack_api.client.toolgroups.list(timeout=5.0)  # Set a reasonable timeout
        
        # Debug the structure of the toolgroups
        if toolgroups:
            # Check the first toolgroup to determine its structure
            first_toolgroup = toolgroups[0] if toolgroups else None
            
            if first_toolgroup:
                # Get the attribute names
                attr_names = dir(first_toolgroup)
                
                # The ToolGroup class has an 'identifier' attribute as per the API definition
                id_attr = 'identifier'
                
                # Filter MCP toolgroups based on the identifier attribute
                mcp_toolgroups = []
                for tg in toolgroups:
                    if hasattr(tg, 'identifier') and isinstance(tg.identifier, str) and tg.identifier.startswith("mcp::"):
                        mcp_toolgroups.append(tg)
                
                if mcp_toolgroups:
                    # Display MCP servers with unregister buttons
                    st.write("Click the button next to a server to unregister it:")
                    
                    for tg in mcp_toolgroups:
                        col1, col2, col3, col4 = st.columns([3, 2, 3, 1])
                        
                        with col1:
                            st.write(f"**{tg.identifier}**")
                        
                        with col2:
                            st.write(tg.provider_id)
                        
                        with col3:
                            if hasattr(tg, 'mcp_endpoint') and tg.mcp_endpoint:
                                st.write(tg.mcp_endpoint.uri)
                            else:
                                st.write("N/A")
                        
                        with col4:
                            # Extract server name from identifier (remove "mcp::" prefix)
                            server_name = tg.identifier.replace("mcp::", "")
                            if st.button("Unregister", key=f"unregister_{server_name}"):
                                try:
                                    # Call the unregister API
                                    llama_stack_api.client.toolgroups.unregister(
                                        toolgroup_id=tg.identifier,
                                        timeout=5.0
                                    )
                                    
                                    # Also clean up provider data
                                    if hasattr(llama_stack_api, "provider_data"):
                                        provider_data = llama_stack_api.provider_data.copy()
                                        
                                        # Remove MCP headers for this server if they exist
                                        if "mcp_headers" in provider_data and hasattr(tg, 'mcp_endpoint') and tg.mcp_endpoint:
                                            if tg.mcp_endpoint.uri in provider_data["mcp_headers"]:
                                                del provider_data["mcp_headers"][tg.mcp_endpoint.uri]
                                        
                                        # Remove server-specific tokens
                                        keys_to_remove = [
                                            f"{server_name}_api_key",
                                            f"{server_name}_token",
                                            f"{server_name}_mcp_token"
                                        ]
                                        for key in keys_to_remove:
                                            if key in provider_data:
                                                del provider_data[key]
                                        
                                        # Update the client with the modified provider data
                                        if hasattr(llama_stack_api, "update_provider_data_dict"):
                                            llama_stack_api.update_provider_data_dict(provider_data)
                                        else:
                                            # Fallback implementation
                                            llama_stack_api.provider_data = provider_data
                                            llama_stack_api.client = LlamaStackClient(
                                                base_url=os.environ.get("LLAMA_STACK_ENDPOINT", "http://localhost:8321"),
                                                provider_data=provider_data,
                                            )
                                    
                                    st.success(f"Successfully unregistered {tg.identifier}")
                                    st.rerun()  # Refresh the page to update the list
                                except Exception as e:
                                    st.error(f"Failed to unregister {tg.identifier}: {str(e)}")
                else:
                    st.info("No MCP toolgroups registered yet.")
            else:
                st.info("No toolgroups found.")
        else:
            st.info("No toolgroups returned from the API.")
    except Exception as e:
        if "timeout" in str(e).lower():
            st.warning("Listing toolgroups timed out. The server might be busy or unreachable.")
        else:
            st.error(f"Failed to list toolgroups: {str(e)}")


if __name__ == "__main__":
    main()
