# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import streamlit as st
from modules.api import llama_stack_api


def providers():
    st.header("🔍 API Providers")
    apis_providers_info = llama_stack_api.client.providers.list()
    # selected_api = st.selectbox("Select an API", list(apis_providers_info.keys()))
    for api in apis_providers_info.keys():
        st.markdown(f"###### {api}")
        st.dataframe([p.to_dict() for p in apis_providers_info[api]], width=500)


providers()
