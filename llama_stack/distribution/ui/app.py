# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import streamlit as st


def main():
    # Evaluation pages
    application_evaluation_page = st.Page(
        "page/evaluations/app_eval.py",
        title="Application Evaluation",
        icon="📊",
        default=False,
    )
    native_evaluation_page = st.Page(
        "page/evaluations/native_eval.py",
        title="Native Evaluation",
        icon="📊",
        default=False,
    )

    # Playground pages
    chat_page = st.Page(
        "page/playground/chat.py", title="Chat", icon="💬", default=True
    )
    rag_page = st.Page("page/playground/rag.py", title="RAG", icon="💬", default=False)

    # Distribution pages
    provider_page = st.Page(
        "page/distribution/providers.py", title="Provider", icon="🔍", default=False
    )
    model_page = st.Page(
        "page/distribution/models.py", title="Models", icon="🔍", default=False
    )
    memory_bank_page = st.Page(
        "page/distribution/memory_banks.py",
        title="Memory Banks",
        icon="🔍",
        default=False,
    )
    shield_page = st.Page(
        "page/distribution/shields.py", title="Shields", icon="🔍", default=False
    )
    scoring_function_page = st.Page(
        "page/distribution/scoring_functions.py",
        title="Scoring Functions",
        icon="🔍",
        default=False,
    )
    eval_task_page = st.Page(
        "page/distribution/eval_tasks.py",
        title="Eval Tasks",
        icon="🔍",
        default=False,
    )

    pg = st.navigation(
        {
            "Playground": [
                chat_page,
                rag_page,
                application_evaluation_page,
                native_evaluation_page,
            ],
            "Inspect": [
                provider_page,
                model_page,
                memory_bank_page,
                shield_page,
                scoring_function_page,
                eval_task_page,
            ],
        },
        expanded=False,
    )
    pg.run()


if __name__ == "__main__":
    main()
