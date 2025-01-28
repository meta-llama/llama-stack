# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.datatypes import Api

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
        "log_probs": [
            "test_completion_log_probs_non_streaming",
            "test_completion_log_probs_streaming",
        ],
    },
    "completion": {
        "streaming": ["test_text_completion_streaming"],
        "non_streaming": ["test_text_completion_non_streaming"],
        "structured_output": ["test_text_completion_structured_output"],
    },
}

VECTORIO_API_TEST_MAP = {
    "retrieve": {
        "": ["test_vector_db_retrieve"],
    }
}

AGENTS_API_TEST_MAP = {
    "create_agent_turn": {
        "rag": ["test_rag_agent"],
        "custom_tool": ["test_custom_tool"],
        "code_execution": ["test_code_interpreter_for_attachments"],
    }
}


API_MAPS = {
    Api.inference: INFERENCE_API_CAPA_TEST_MAP,
    Api.vector_io: VECTORIO_API_TEST_MAP,
    Api.agents: AGENTS_API_TEST_MAP,
}
