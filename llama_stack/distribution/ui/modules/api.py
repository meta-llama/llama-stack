# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import streamlit as st

from llama_stack_client import LlamaStackClient


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
            
        # Initialize the client
        self.client = LlamaStackClient(
            base_url=os.environ.get("LLAMA_STACK_ENDPOINT", "http://localhost:8321"),
            provider_data=self.provider_data,
        )

    def update_provider_data(self, key, value):
        """Update a specific provider data key and reinitialize the client"""
        self.provider_data[key] = value
        
        # Reinitialize the client with updated provider data
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
