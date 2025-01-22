# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


INFERENCE_API_CAPA_TEST_MAP = {
    "chat_completion": {
        "streaming": [
            "test_text_chat_completion_streaming",
            "test_image_chat_completion_streaming",
        ],
        "non_streaming": [
            "test_image_chat_completion_non_streaming",
            "test_text_chat_completion_non_streaming",
        ],
        "tool_calling": [
            "test_text_chat_completion_with_tool_calling_and_streaming",
            "test_text_chat_completion_with_tool_calling_and_non_streaming",
        ],
    },
    "completion": {
        "streaming": ["test_text_completion_streaming"],
        "non_streaming": ["test_text_completion_non_streaming"],
        "structured_output": ["test_text_completion_structured_output"],
    },
}

MEMORY_API_TEST_MAP = {
    "/insert, /query": {
        "inline": ["test_memory_bank_insert_inline_and_query"],
        "url": ["test_memory_bank_insert_from_url_and_query"],
    }
}

AGENTS_API_TEST_MAP = {
    "create_agent_turn": {
        "rag": ["test_rag_agent"],
        "custom_tool": ["test_custom_tool"],
        "code_execution": ["test_code_execution"],
    }
}


API_MAPS = {
    "inference": INFERENCE_API_CAPA_TEST_MAP,
    "memory": MEMORY_API_TEST_MAP,
    "agents": AGENTS_API_TEST_MAP,
}