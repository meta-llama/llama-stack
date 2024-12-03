# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import streamlit as st
from modules.api import llama_stack_api

st.header("Memory Banks")
memory_banks_info = {
    m.identifier: m.to_dict() for m in llama_stack_api.client.memory_banks.list()
}

if len(memory_banks_info) > 0:
    selected_memory_bank = st.selectbox(
        "Select a memory bank", list(memory_banks_info.keys())
    )
    st.json(memory_banks_info[selected_memory_bank])
else:
    st.info("No memory banks found")
