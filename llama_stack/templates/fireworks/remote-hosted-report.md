# Report for fireworks distribution

## Supported Models
| Model Descriptor | fireworks |
|:---|:---|
| meta-llama/Llama-3-8B-Instruct | ❌ |
| meta-llama/Llama-3-70B-Instruct | ❌ |
| meta-llama/Llama-3.1-8B-Instruct | ❌ |
| meta-llama/Llama-3.1-70B-Instruct | ❌ |
| meta-llama/Llama-3.1-405B-Instruct-FP8 | ❌ |
| meta-llama/Llama-3.2-1B-Instruct | ❌ |
| meta-llama/Llama-3.2-3B-Instruct | ❌ |
| meta-llama/Llama-3.2-11B-Vision-Instruct | ❌ |
| meta-llama/Llama-3.2-90B-Vision-Instruct | ❌ |
| meta-llama/Llama-3.3-70B-Instruct | ❌ |
| meta-llama/Llama-Guard-3-11B-Vision | ❌ |
| meta-llama/Llama-Guard-3-1B | ❌ |
| meta-llama/Llama-Guard-3-8B | ❌ |
| meta-llama/Llama-Guard-2-8B | ❌ |

## Inference
| Model | API | Capability | Test | Status |
|:----- |:-----|:-----|:-----|:-----|
| Text | /chat_completion | streaming | test_text_chat_completion_streaming | ❌ |
| Vision | /chat_completion | streaming | test_image_chat_completion_streaming | ❌ |
| Vision | /chat_completion | non_streaming | test_image_chat_completion_non_streaming | ❌ |
| Text | /chat_completion | non_streaming | test_text_chat_completion_non_streaming | ❌ |
| Text | /chat_completion | tool_calling | test_text_chat_completion_with_tool_calling_and_streaming | ❌ |
| Text | /chat_completion | tool_calling | test_text_chat_completion_with_tool_calling_and_non_streaming | ❌ |
| Text | /completion | streaming | test_text_completion_streaming | ❌ |
| Text | /completion | non_streaming | test_text_completion_non_streaming | ❌ |
| Text | /completion | structured_output | test_text_completion_structured_output | ❌ |

## Memory:
| API | Capability | Test | Status |
|:-----|:-----|:-----|:-----|
| /insert, /query | inline | test_memory_bank_insert_inline_and_query | ❌ |
| /insert, /query | url | test_memory_bank_insert_from_url_and_query | ❌ |

## Agents
| API | Capability | Test | Status |
|:-----|:-----|:-----|:-----|
| create_agent_turn | rag | test_rag_agent | ❌ |
| create_agent_turn | custom_tool | test_custom_tool | ❌ |
| create_agent_turn | code_execution | test_code_execution | ❌ |
