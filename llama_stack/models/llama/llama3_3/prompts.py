# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import textwrap
from typing import List

from llama_models.datatypes import (
    BuiltinTool,
    RawMessage,
    StopReason,
    ToolCall,
    ToolPromptFormat,
)

from ..prompt_format import (
    # llama3_1_e2e_tool_call_dialog,
    TextCompletionContent,
    UseCase,
    llama3_1_builtin_tool_call_dialog,
    llama3_1_custom_tool_call_dialog,
)


def wolfram_alpha_response():
    return textwrap.dedent(
        """
        {
            "queryresult": {
                "success": true,
                "inputstring": "100th decimal of pi",
                "pods": [
                    {
                        "title": "Input interpretation",
                        "subpods": [
                            {
                                "title": "",
                                "plaintext": "100th digit | \u03c0"
                            }
                        ]
                    },
                    {
                        "title": "Nearby digits",
                        "subpods": [
                            {
                                "title": "",
                                "plaintext": "...86208998628034825342117067982148086513282306647093..."
                            }
                        ]
                    },
                    {
                        "title": "Result",
                        "primary": true,
                        "subpods": [
                            {
                                "title": "",
                                "plaintext": "7"
                            }
                        ]
                    }
                ]
            }
        }
        """
    )


def usecases() -> List[UseCase | str]:
    return [
        textwrap.dedent(
            """
            # Llama 3.1 - Prompt Formats
            ## Tokens
            Here is a list of special tokens that are supported by Llama 3.1:
            - `<|begin_of_text|>`: Specifies the start of the prompt
            - `<|end_of_text|>`: Model will cease to generate more tokens. This token is generated only by the base models.
            - `<|finetune_right_pad_id|>`: This token is used for padding text sequences to the same length in a batch.
            - `<|start_header_id|>` and `<|end_header_id|>`: These tokens enclose the role for a particular message. The possible roles are: [system, user, assistant and tool]
            - `<|eom_id|>`: End of message. A message represents a possible stopping point for execution where the model can inform the executor that a tool call needs to be made. This is used for multi-step interactions between the model and any available tools. This token is emitted by the model when the Environment: ipython instruction is used in the system prompt, or if the model calls for a built-in tool.
            - `<|eot_id|>`: End of turn. Represents when the model has determined that it has finished interacting with the user message that initiated its response. This is used in two scenarios:
                - at the end of a direct interaction between the model and the user
                - at the end of multiple interactions between the model and any available tools
                This token signals to the executor that the model has finished generating a response.
            - `<|python_tag|>`: Is a special tag used in the model's response to signify a tool call.
            """
        ),
        textwrap.dedent(
            """
            There are 4 different roles that are supported by Llama 3.1
            - `system`: Sets the context in which to interact with the AI model. It typically includes rules, guidelines, or necessary information that helps the model respond effectively.
            - `user`: Represents the human interacting with the model. It includes the inputs, commands, and questions to the model.
            - `tool`: A new role introduced in Llama 3.1. This role is used to mark messages with the output of a tool call when sent back to the model from the executor. (The actual token used by the model for this role is "ipython".)
            - `assistant`: Represents the response generated by the AI model based on the context provided in the `system`, `tool` and `user` prompts.
            """
        ),
        UseCase(
            title="Llama 3.1 Base Model",
            description="Text completion for Llama 3.1 base model uses this format.",
            dialogs=[TextCompletionContent(content="Color of sky is blue but sometimes can also be")],
            notes="Note start special tag",
        ),
        "## Llama 3.1 Instruct Model",
        UseCase(
            title="User and assistant conversation",
            description="Here is a regular multi-turn user assistant conversation and how its formatted.",
            dialogs=[
                [
                    RawMessage(role="system", content="You are a helpful assistant"),
                    RawMessage(
                        role="user",
                        content="Answer who are you in the form of jeopardy?",
                    ),
                ]
            ],
            notes="",
        ),
        "## Tool Calling Formats",
        textwrap.dedent(
            """
            The three built-in tools (brave_search, wolfram_alpha, and code interpreter) can be turned on using the system prompt:
            - Brave Search: Tool call to perform web searches.
            - Wolfram Alpha: Tool call to perform complex mathematical calculations.
            - Code Interpreter: Enables the model to output python code.
            """
        ),
        UseCase(
            title="Builtin Tool Calling",
            description=textwrap.dedent(
                """
                Here is an example of a conversation using brave search
                """
            ),
            dialogs=[llama3_1_builtin_tool_call_dialog()],
            notes=textwrap.dedent(
                """
                - Just including Environment: ipython turns on code interpreter; therefore, you don't need to specify code interpretation on the Tools: line. The model can generate python code which is interpreted by the executor, with the result provided back to the model.
                - The message body of the assistant response starts with a special tag <|python_tag|>
                - As alluded to above, in such an environment, the model can generate <|eom_id|> instead of just the standard <|eot_id|> . The latter indicates the turn is finished, while the former indicates continued multi-step reasoning. That is, the model is expecting a continuation message with the output of the tool call.
                - The model tool call response is of the form `tool.call(query="...")` wher tool is `brave_search` or `wolfram_alpha`
                """
            ),
        ),
        UseCase(
            title="Builtin Code Interpreter",
            description="Here is an actual example of model responding with code",
            dialogs=[
                [
                    RawMessage(role="system", content="Environment: ipython"),
                    RawMessage(
                        role="user",
                        content="Write code to check if number is prime, use that to see if the number 7 is prime",
                    ),
                ],
            ],
            notes=textwrap.dedent(
                """
                - Model starts with <|python_tag|> and continues writing python code that it needs to be executed
                - No explicit mention of code_interpreter in system prompt. `Environment: ipython` implicitly enables it.
                """
            ),
        ),
        UseCase(
            title="Built-in tools full interaction",
            description="Here is a full interaction with the built-in tools including the tool response and the final assistant response.",
            dialogs=[
                [
                    RawMessage(
                        role="system",
                        content="Environment: ipython\nTools: brave_search, wolfram_alpha\n",
                    ),
                    RawMessage(role="user", content="What is the 100th decimal of pi?"),
                    RawMessage(
                        content="",
                        stop_reason=StopReason.end_of_message,
                        tool_calls=[
                            ToolCall(
                                call_id="tool_call_id",
                                tool_name=BuiltinTool.wolfram_alpha,
                                arguments={"query": "100th decimal of pi"},
                            )
                        ],
                    ),
                    RawMessage(
                        role="tool",
                        content=wolfram_alpha_response(),
                    ),
                ],
            ],
            notes=textwrap.dedent(
                """
                - Note the `<|python_tag|>` in the assistant response.
                - Role is `tool` for the wolfram alpha response that is passed back to the model.
                - Final message from assistant has <|eot_id|> tag.
                """
            ),
        ),
        "## Zero shot tool calling",
        UseCase(
            title="JSON based tool calling",
            description=textwrap.dedent(
                """
                Llama models can now output custom tool calls from a single message to allow easier tool calling.
                The following prompts provide an example of how custom tools can be called from the output of the model.
                It's important to note that the model itself does not execute the calls; it provides structured output to facilitate calling by an executor.
                """
            ),
            dialogs=[llama3_1_custom_tool_call_dialog()],
            notes=textwrap.dedent(
                """
                - JSON format for providing tools needs name, description and parameters
                - Model responds with `<|python_tag|>` and `<|eom_id|>` as `Environment: ipython` was in the system prompt
                - Instructions for tools added as a user message
                - Only single tool calls are supported as of now
                """
            ),
        ),
        # FIXME: This is not working yet as expected
        # UseCase(
        #     title="E2E tool call example",
        #     description=textwrap.dedent(
        #         """
        #         Here is an example showing the whole multi-step turn by taking custom tool outputs and passing back to the model.
        #         """
        #     ),
        #     dialogs=[
        #         llama3_1_e2e_tool_call_dialog(
        #             tool_prompt_format=ToolPromptFormat.function_tag
        #         )
        #     ],
        #     notes="",
        # ),
        "## Example of a user defined tool calling",
        UseCase(
            title="`<function>` based tool calling",
            description=textwrap.dedent(
                """
                Here is an example of how you could also write custom instructions for model to do zero shot tool calling.
                In this example, we define a custom tool calling format using the `<function>` tag.
                """
            ),
            dialogs=[llama3_1_custom_tool_call_dialog(ToolPromptFormat.function_tag)],
            notes=textwrap.dedent(
                """
                - In this case, model does NOT respond with `<|python_tag|>` and ends with `<|eot_id|>`
                - Instructions for tools added as a user message
                """
            ),
        ),
    ]
