# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import streamlit as st
from modules.api import llama_stack_api

# Eval Tasks Section
st.header("Eval Tasks")

eval_tasks_info = {
    d.identifier: d.to_dict() for d in llama_stack_api.client.eval_tasks.list()
}

selected_eval_task = st.selectbox("Select an eval task", list(eval_tasks_info.keys()))
st.json(eval_tasks_info[selected_eval_task], expanded=True)
