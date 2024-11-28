# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import streamlit as st
from modules.api import llama_stack_api

st.title("🔍 Inspect Distribution")

# API Section
st.header("API Providers")
apis_providers_info = llama_stack_api.client.providers.list()
selected_api = st.selectbox("Select an API", list(apis_providers_info.keys()))
st.dataframe([p.to_dict() for p in apis_providers_info[selected_api]], width=500)

# for api in apis_providers_info:
#     st.write(api)
#     st.dataframe([p.to_dict() for p in apis_providers_info[api]], width=500)

# Models Section
st.header("Models")
models_info = {m.identifier: m.to_dict() for m in llama_stack_api.client.models.list()}

selected_model = st.selectbox("Select a model", list(models_info.keys()))
st.json(models_info[selected_model])

# Memory Banks Section
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

# Shields Section
st.header("Shields")

shields_info = {
    s.identifier: s.to_dict() for s in llama_stack_api.client.shields.list()
}

selected_shield = st.selectbox("Select a shield", list(shields_info.keys()))
st.json(shields_info[selected_shield])

# Datasets Section
st.header("Datasets")

datasets_info = {
    d.identifier: d.to_dict() for d in llama_stack_api.client.datasets.list()
}

selected_dataset = st.selectbox("Select a dataset", list(datasets_info.keys()))
st.json(datasets_info[selected_dataset], expanded=False)

# Scoring Functions Section
st.header("Scoring Functions")

scoring_functions_info = {
    s.identifier: s.to_dict() for s in llama_stack_api.client.scoring_functions.list()
}

selected_scoring_function = st.selectbox(
    "Select a scoring function", list(scoring_functions_info.keys())
)
st.json(scoring_functions_info[selected_scoring_function], expanded=False)

# Eval Tasks Section
st.header("Eval Tasks")

eval_tasks_info = {
    d.identifier: d.to_dict() for d in llama_stack_api.client.eval_tasks.list()
}

selected_eval_task = st.selectbox("Select an eval task", list(eval_tasks_info.keys()))
st.json(eval_tasks_info[selected_eval_task], expanded=True)
