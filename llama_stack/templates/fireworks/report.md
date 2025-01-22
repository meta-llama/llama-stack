# Report for fireworks distribution

## Supported Models:
| Model Descriptor | fireworks |
|:---|:---|
| Llama-3-8B-Instruct | ❌ |
| Llama-3-70B-Instruct | ❌ |
| Llama3.1-8B-Instruct | ✅ |
| Llama3.1-70B-Instruct | ✅ |
| Llama3.1-405B-Instruct | ✅ |
| Llama3.1-405B-Instruct | ✅ |
| Llama3.1-405B-Instruct | ✅ |
| Llama3.2-1B-Instruct | ✅ |
| Llama3.2-3B-Instruct | ✅ |
| Llama3.2-1B-Instruct | ✅ |
| Llama3.2-1B-Instruct | ✅ |
| Llama3.2-3B-Instruct | ✅ |
| Llama3.2-3B-Instruct | ✅ |
| Llama3.2-11B-Vision-Instruct | ✅ |
| Llama3.2-90B-Vision-Instruct | ✅ |
| Llama3.3-70B-Instruct | ✅ |
| Llama-Guard-3-11B-Vision | ✅ |
| Llama-Guard-3-1B | ❌ |
| Llama-Guard-3-1B | ❌ |
| Llama-Guard-3-8B | ✅ |
| Llama-Guard-3-8B | ✅ |
| Llama-Guard-2-8B | ❌ |

## Inference:
| Model | API | Capability | Test | Status |
|:----- |:-----|:-----|:-----|:-----|
| Text | /chat_completion | streaming | test_text_chat_completion_streaming | Passed |
| Vision | /chat_completion | streaming | test_image_chat_completion_streaming | Passed |
| Text | /chat_completion | non_streaming | test_text_chat_completion_non_streaming | Passed |
| Vision | /chat_completion | non_streaming | test_image_chat_completion_non_streaming | Passed |
| Text | /chat_completion | tool_calling | test_text_chat_completion_with_tool_calling_and_streaming | Passed |
| Text | /chat_completion | tool_calling | test_text_chat_completion_with_tool_calling_and_non_streaming | Passed |
| Text | /completion | streaming | test_text_completion_streaming | Passed |
| Text | /completion | non_streaming | test_text_completion_non_streaming | Passed |
| Text | /completion | structured_output | test_text_completion_structured_output | Passed |

## Memory:
| API | Capability | Test | Status |
|:-----|:-----|:-----|:-----|
| insert and query | inline | test_memory_bank_insert_inline_and_query | Error |
| insert and query | url | test_memory_bank_insert_from_url_and_query | Failed |

## Agent:
| API | Capability | Test | Status |
|:-----|:-----|:-----|:-----|
| create_agent_turn | rag | test_rag_agent | Failed |
| create_agent_turn | custom_tool | test_custom_tool | Passed |
| create_agent_turn | code_execution | test_code_execution | Failed |
