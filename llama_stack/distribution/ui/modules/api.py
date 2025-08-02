# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os
import streamlit as st

from llama_stack_client import LlamaStackClient


# Constants
MCP_CONFIG_DIR = os.path.expanduser("~/.llama_stack/mcp_servers")
MCP_CONFIG_FILE = os.path.join(MCP_CONFIG_DIR, "config.json")

class LlamaStackApi:
    def __init__(self):
        # Initialize provider data from environment variables
        self.provider_data = {
            "fireworks_api_key": os.environ.get("FIREWORKS_API_KEY", ""),
            "together_api_key": os.environ.get("TOGETHER_API_KEY", ""),
            "sambanova_api_key": os.environ.get("SAMBANOVA_API_KEY", ""),
            "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
            "tavily_search_api_key": os.environ.get("TAVILY_SEARCH_API_KEY", ""),
        }
        
        # Check if we have any API keys stored in session state
        if st.session_state.get("tavily_search_api_key"):
            self.provider_data["tavily_search_api_key"] = st.session_state.get("tavily_search_api_key")
        
        # Load MCP server configurations
        self.mcp_servers = self.load_mcp_config()
            
        # Initialize the client
        try:
            # Try to initialize with MCP servers support
            self.client = LlamaStackClient(
                base_url=os.environ.get("LLAMA_STACK_ENDPOINT", "http://localhost:8321"),
                provider_data=self.provider_data,
                mcp_servers=self.mcp_servers.get("servers", {}),
            )
        except TypeError:
            # Fall back to initialization without MCP servers if not supported
            self.client = LlamaStackClient(
                base_url=os.environ.get("LLAMA_STACK_ENDPOINT", "http://localhost:8321"),
                provider_data=self.provider_data,
            )
            st.warning("MCP servers support not available in the current LlamaStackClient version.")

    def load_mcp_config(self):
        """Load MCP server configurations from file."""
        if os.path.exists(MCP_CONFIG_FILE):
            try:
                with open(MCP_CONFIG_FILE, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                st.error("Error loading MCP server configuration file.")
        
        # Default empty configuration
        return {
            "servers": {},
            "inputs": []
        }
    
    def update_provider_data(self, key, value):
        """Update a specific provider data key and reinitialize the client"""
        self.provider_data[key] = value
        self.update_provider_data_dict(self.provider_data)
    
    def update_provider_data_dict(self, provider_data):
        """Update the provider data dictionary and reinitialize the client"""
        self.provider_data = provider_data
        
        # Reinitialize the client with updated provider data
        try:
            # Try to initialize with MCP servers support
            self.client = LlamaStackClient(
                base_url=os.environ.get("LLAMA_STACK_ENDPOINT", "http://localhost:8321"),
                provider_data=self.provider_data,
                mcp_servers=self.mcp_servers.get("servers", {}),
            )
        except TypeError:
            # Fall back to initialization without MCP servers if not supported
            self.client = LlamaStackClient(
                base_url=os.environ.get("LLAMA_STACK_ENDPOINT", "http://localhost:8321"),
                provider_data=self.provider_data,
            )
    
    def update_mcp_servers(self):
        """Reload MCP server configurations and reinitialize the client"""
        self.mcp_servers = self.load_mcp_config()
        
        # Reinitialize the client with updated MCP servers
        try:
            # Try to initialize with MCP servers support
            self.client = LlamaStackClient(
                base_url=os.environ.get("LLAMA_STACK_ENDPOINT", "http://localhost:8321"),
                provider_data=self.provider_data,
                mcp_servers=self.mcp_servers.get("servers", {}),
            )
        except TypeError:
            # Fall back to initialization without MCP servers if not supported
            self.client = LlamaStackClient(
                base_url=os.environ.get("LLAMA_STACK_ENDPOINT", "http://localhost:8321"),
                provider_data=self.provider_data,
            )

    def run_scoring(self, row, scoring_function_ids: list[str], scoring_params: dict | None):
        """Run scoring on a single row"""
        if not scoring_params:
            scoring_params = dict.fromkeys(scoring_function_ids)
        return self.client.scoring.score(input_rows=[row], scoring_functions=scoring_params)


llama_stack_api = LlamaStackApi()
