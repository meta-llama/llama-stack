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
        icon="🦙",
        default=False,
    )

    # Playground pages
    chat_page = st.Page(
        "page/playground/chat.py", title="Chat", icon="💬", default=True
    )

    pg = st.navigation(
        {"Evaluations": [application_evaluation_page], "Playground": [chat_page]}
    )
    pg.run()


if __name__ == "__main__":
    main()
