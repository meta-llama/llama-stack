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
import unittest
from datetime import datetime

from .prompt_templates import (
    BuiltinToolGenerator,
    FunctionTagCustomToolGenerator,
    JsonCustomToolGenerator,
    PythonListCustomToolGenerator,
    SystemDefaultGenerator,
)


class PromptTemplateTests(unittest.TestCase):
    def check_generator_output(self, generator, expected_text):
        example = generator.data_examples()[0]

        pt = generator.gen(example)
        text = pt.render()
        # print(text)  # debugging
        assert text == expected_text, f"Expected:\n{expected_text}\nActual:\n{text}"

    def test_system_default(self):
        generator = SystemDefaultGenerator()
        today = datetime.now().strftime("%d %B %Y")
        expected_text = f"Cutting Knowledge Date: December 2023\nToday Date: {today}"
        self.check_generator_output(generator, expected_text)

    def test_system_builtin_only(self):
        generator = BuiltinToolGenerator()
        expected_text = textwrap.dedent(
            """
            Environment: ipython
            Tools: brave_search, wolfram_alpha
            """
        )
        self.check_generator_output(generator, expected_text.strip("\n"))

    def test_system_custom_only(self):
        self.maxDiff = None
        generator = JsonCustomToolGenerator()
        expected_text = textwrap.dedent(
            """
            Answer the user's question by making use of the following functions if needed.
            If none of the function can be used, please say so.
            Here is a list of functions in JSON format:
            {
                "type": "function",
                "function": {
                    "name": "trending_songs",
                    "description": "Returns the trending songs on a Music site",
                    "parameters": {
                        "type": "object",
                        "properties": [
                            {
                                "n": {
                                    "type": "object",
                                    "description": "The number of songs to return"
                                }
                            },
                            {
                                "genre": {
                                    "type": "object",
                                    "description": "The genre of the songs to return"
                                }
                            }
                        ],
                        "required": ["n"]
                    }
                }
            }

            Return function calls in JSON format.
            """
        )
        self.check_generator_output(generator, expected_text.strip("\n"))

    def test_system_custom_function_tag(self):
        self.maxDiff = None
        generator = FunctionTagCustomToolGenerator()
        expected_text = textwrap.dedent(
            """
            You have access to the following functions:

            Use the function 'trending_songs' to 'Returns the trending songs on a Music site':
            {"name": "trending_songs", "description": "Returns the trending songs on a Music site", "parameters": {"genre": {"description": "The genre of the songs to return", "param_type": "str", "required": false}, "n": {"description": "The number of songs to return", "param_type": "int", "required": true}}}

            Think very carefully before calling functions.
            If you choose to call a function ONLY reply in the following format with no prefix or suffix:

            <function=example_function_name>{"example_name": "example_value"}</function>

            Reminder:
            - If looking for real time information use relevant functions before falling back to brave_search
            - Function calls MUST follow the specified format, start with <function= and end with </function>
            - Required parameters MUST be specified
            - Only call one function at a time
            - Put the entire function call reply on one line
            """
        )
        self.check_generator_output(generator, expected_text.strip("\n"))

    def test_llama_3_2_system_zero_shot(self):
        generator = PythonListCustomToolGenerator()
        expected_text = textwrap.dedent(
            """
            You are an expert in composing functions. You are given a question and a set of possible functions.
            Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
            If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
            also point it out. You should only return the function call in tools call sections.

            If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
            You SHOULD NOT include any other text in the response.

            Here is a list of functions in JSON format that you can invoke.

            [
                {
                    "name": "get_weather",
                    "description": "Get weather info for places",
                    "parameters": {
                        "type": "dict",
                        "required": ["city"],
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The name of the city to get the weather for"
                            },
                            "metric": {
                                "type": "string",
                                "description": "The metric for weather. Options are: celsius, fahrenheit",
                                "default": "celsius"
                            }
                        }
                    }
                }
            ]
            """
        )
        self.check_generator_output(generator, expected_text.strip("\n"))

    def test_llama_3_2_provided_system_prompt(self):
        generator = PythonListCustomToolGenerator()
        expected_text = textwrap.dedent(
            """
            Overriding message.

            If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
            You SHOULD NOT include any other text in the response.

            Here is a list of functions in JSON format that you can invoke.

            [
                {
                    "name": "get_weather",
                    "description": "Get weather info for places",
                    "parameters": {
                        "type": "dict",
                        "required": ["city"],
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The name of the city to get the weather for"
                            },
                            "metric": {
                                "type": "string",
                                "description": "The metric for weather. Options are: celsius, fahrenheit",
                                "default": "celsius"
                            }
                        }
                    }
                }
            ]"""
        )
        user_system_prompt = textwrap.dedent(
            """
            Overriding message.

            {{ function_description }}
            """
        )
        example = generator.data_examples()[0]

        pt = generator.gen(example, user_system_prompt)
        text = pt.render()
        assert text == expected_text, f"Expected:\n{expected_text}\nActual:\n{text}"
