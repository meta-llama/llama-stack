# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import streamlit as st

from llama_stack.distribution.ui.modules.api import llama_stack_api


def providers():
    st.header("🔍 API Providers")
    
    # API Key Management Section
    st.subheader("API Key Management")
    
    # Create a form for API key input
    with st.form("api_keys_form"):
        # Get the current value from session state or environment variable
        tavily_key = st.session_state.get("tavily_search_api_key", os.environ.get("TAVILY_SEARCH_API_KEY", ""))
        
        # Input field for Tavily Search API key
        tavily_search_api_key = st.text_input(
            "Tavily Search API Key", 
            value=tavily_key,
            type="password",
            help="Enter your Tavily Search API key. This will be used for search operations."
        )
        
        # Submit button
        submit_button = st.form_submit_button("Save API Keys")
        
        if submit_button:
            # Store the API key in session state
            st.session_state["tavily_search_api_key"] = tavily_search_api_key
            
            # Update the client with the new API key
            llama_stack_api.update_provider_data("tavily_search_api_key", tavily_search_api_key)
            
            st.success("API keys saved successfully!")
    
    # Display API Providers
    st.subheader("Available API Providers")
    apis_providers_lst = llama_stack_api.client.providers.list()
    api_to_providers = {}
    for api_provider in apis_providers_lst:
        if api_provider.api in api_to_providers:
            api_to_providers[api_provider.api].append(api_provider)
        else:
            api_to_providers[api_provider.api] = [api_provider]

    for api in api_to_providers.keys():
        st.markdown(f"###### {api}")
        st.dataframe([x.to_dict() for x in api_to_providers[api]], width=500)


providers()
