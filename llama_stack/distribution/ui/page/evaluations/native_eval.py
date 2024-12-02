# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import streamlit as st

from modules.api import llama_stack_api


def native_evaluation_page():

    st.set_page_config(page_title="Native Evaluations", page_icon="🦙")
    st.title("🦙 Llama Stack Native Evaluations")

    # Select Eval Tasks
    st.subheader("Select Eval Tasks")
    eval_tasks = llama_stack_api.client.eval_tasks.list()
    eval_tasks = {et.identifier: et for et in eval_tasks}
    eval_tasks_names = list(eval_tasks.keys())
    selected_eval_task = st.selectbox(
        "Choose an eval task.",
        options=eval_tasks_names,
        help="Choose an eval task. Each eval task is parameterized by a dataset, and list of scoring functions.",
    )
    st.json(eval_tasks[selected_eval_task], expanded=True)

    # Define Eval Candidate
    st.subheader("Define Eval Candidate")
    # eval_candidate = {}


native_evaluation_page()
