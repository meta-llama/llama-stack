### Fireworks
|                            filepath                            |                              function                              | passed | SUBTOTAL |
| -------------------------------------------------------------- | ------------------------------------------------------------------ | -----: | -------: |
| llama_stack/providers/tests/inference/test_text_inference.py   | TestInference.test_completion                                      |      1 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py   | TestInference.test_chat_completion_non_streaming                   |      1 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py   | TestInference.test_structured_output                               |      1 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py   | TestInference.test_chat_completion_streaming                       |      1 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py   | TestInference.test_chat_completion_with_tool_calling               |      1 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py   | TestInference.test_chat_completion_with_tool_calling_streaming     |      1 |        1 |
| llama_stack/providers/tests/inference/test_vision_inference.py | TestVisionModelInference.test_vision_chat_completion_non_streaming |      2 |        2 |
| llama_stack/providers/tests/inference/test_vision_inference.py | TestVisionModelInference.test_vision_chat_completion_streaming     |      1 |        1 |
| TOTAL                                                          |                                                                    |      9 |        9 |



### Together
|                            filepath                            |                              function                              | passed | SUBTOTAL |
| -------------------------------------------------------------- | ------------------------------------------------------------------ | -----: | -------: |
| llama_stack/providers/tests/inference/test_text_inference.py   | TestInference.test_completion                                      |      1 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py   | TestInference.test_chat_completion_non_streaming                   |      1 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py   | TestInference.test_structured_output                               |      1 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py   | TestInference.test_chat_completion_streaming                       |      1 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py   | TestInference.test_chat_completion_with_tool_calling               |      1 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py   | TestInference.test_chat_completion_with_tool_calling_streaming     |      1 |        1 |
| llama_stack/providers/tests/inference/test_vision_inference.py | TestVisionModelInference.test_vision_chat_completion_non_streaming |      2 |        2 |
| llama_stack/providers/tests/inference/test_vision_inference.py | TestVisionModelInference.test_vision_chat_completion_streaming     |      1 |        1 |
| TOTAL                                                          |                                                                    |      9 |        9 |


### vLLM

|                           filepath                           |                            function                            | passed | skipped | SUBTOTAL |
| ------------------------------------------------------------ | -------------------------------------------------------------- | -----: | ------: | -------: |
| llama_stack/providers/tests/inference/test_text_inference.py | TestInference.test_model_list                                  |      1 |       0 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py | TestInference.test_chat_completion_non_streaming               |      1 |       0 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py | TestInference.test_structured_output                           |      1 |       0 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py | TestInference.test_chat_completion_streaming                   |      1 |       0 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py | TestInference.test_chat_completion_with_tool_calling           |      1 |       0 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py | TestInference.test_chat_completion_with_tool_calling_streaming |      1 |       0 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py | TestInference.test_completion                                  |      0 |       1 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py | TestInference.test_completion_logprobs                         |      0 |       1 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py | TestInference.test_completion_structured_output                |      0 |       1 |        1 |
| TOTAL                                                        |                                                                |      6 |       3 |        9 |

### Ollama
|                           filepath                           |                            function                            | passed | SUBTOTAL |
| ------------------------------------------------------------ | -------------------------------------------------------------- | -----: | -------: |
| llama_stack/providers/tests/inference/test_text_inference.py | TestInference.test_completion                                  |      1 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py | TestInference.test_chat_completion_non_streaming               |      1 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py | TestInference.test_structured_output                           |      1 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py | TestInference.test_chat_completion_streaming                   |      1 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py | TestInference.test_chat_completion_with_tool_calling           |      1 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py | TestInference.test_chat_completion_with_tool_calling_streaming |      1 |        1 |
| TOTAL                                                        |                                                                |      6 |        6 |


### tgi

|                     filepath                     |                            function                            | passed | skipped | SUBTOTAL |
| ------------------------------------------------ | -------------------------------------------------------------- | -----: | ------: | -------: |
| providers/tests/inference/test_text_inference.py | TestInference.test_model_list                                  |      1 |       0 |        1 |
| providers/tests/inference/test_text_inference.py | TestInference.test_completion                                  |      1 |       0 |        1 |
| providers/tests/inference/test_text_inference.py | TestInference.test_chat_completion_non_streaming               |      1 |       0 |        1 |
| providers/tests/inference/test_text_inference.py | TestInference.test_structured_output                           |      1 |       0 |        1 |
| providers/tests/inference/test_text_inference.py | TestInference.test_chat_completion_streaming                   |      1 |       0 |        1 |
| providers/tests/inference/test_text_inference.py | TestInference.test_chat_completion_with_tool_calling           |      1 |       0 |        1 |
| providers/tests/inference/test_text_inference.py | TestInference.test_chat_completion_with_tool_calling_streaming |      1 |       0 |        1 |
| providers/tests/inference/test_text_inference.py | TestInference.test_completion_logprobs                         |      0 |       1 |        1 |
| providers/tests/inference/test_text_inference.py | TestInference.test_completion_structured_output                |      0 |       1 |        1 |
| TOTAL                                            |                                                                |      7 |       2 |        9 |


### vLLM

|                           filepath                           |                            function                            | passed | skipped | SUBTOTAL |
| ------------------------------------------------------------ | -------------------------------------------------------------- | -----: | ------: | -------: |
| llama_stack/providers/tests/inference/test_text_inference.py | TestInference.test_model_list                                  |      1 |       0 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py | TestInference.test_chat_completion_non_streaming               |      1 |       0 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py | TestInference.test_structured_output                           |      1 |       0 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py | TestInference.test_chat_completion_streaming                   |      1 |       0 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py | TestInference.test_chat_completion_with_tool_calling           |      1 |       0 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py | TestInference.test_chat_completion_with_tool_calling_streaming |      1 |       0 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py | TestInference.test_completion                                  |      0 |       1 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py | TestInference.test_completion_logprobs                         |      0 |       1 |        1 |
| llama_stack/providers/tests/inference/test_text_inference.py | TestInference.test_completion_structured_output                |      0 |       1 |        1 |
| TOTAL                                                        |                                                                |      6 |       3 |        9 |
