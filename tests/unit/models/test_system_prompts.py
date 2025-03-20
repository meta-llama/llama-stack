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

from llama_stack.models.llama.llama3.prompt_templates import (
    BuiltinToolGenerator,
    FunctionTagCustomToolGenerator,
    JsonCustomToolGenerator,
    PythonListCustomToolGenerator,
    SystemDefaultGenerator,
)


class PromptTemplateTests(unittest.TestCase):
    def check_generator_output(self, generator):
        for example in generator.data_examples():
            pt = generator.gen(example)
            text = pt.render()
            # print(text)  # debugging
            if not example:
                continue
            for tool in example:
                assert tool.tool_name in text

    def test_system_default(self):
        generator = SystemDefaultGenerator()
        today = datetime.now().strftime("%d %B %Y")
        expected_text = f"Cutting Knowledge Date: December 2023\nToday Date: {today}"
        assert expected_text.strip("\n") == generator.gen(generator.data_examples()[0]).render()

    def test_system_builtin_only(self):
        generator = BuiltinToolGenerator()
        expected_text = textwrap.dedent(
            """
            Environment: ipython
            Tools: brave_search, wolfram_alpha
            """
        )
        assert expected_text.strip("\n") == generator.gen(generator.data_examples()[0]).render()

    def test_system_custom_only(self):
        self.maxDiff = None
        generator = JsonCustomToolGenerator()
        self.check_generator_output(generator)

    def test_system_custom_function_tag(self):
        self.maxDiff = None
        generator = FunctionTagCustomToolGenerator()
        self.check_generator_output(generator)

    def test_llama_3_2_system_zero_shot(self):
        generator = PythonListCustomToolGenerator()
        self.check_generator_output(generator)

    def test_llama_3_2_provided_system_prompt(self):
        generator = PythonListCustomToolGenerator()
        user_system_prompt = textwrap.dedent(
            """
            Overriding message.

            {{ function_description }}
            """
        )
        example = generator.data_examples()[0]

        pt = generator.gen(example, user_system_prompt)
        text = pt.render()
        assert "Overriding message." in text
        assert '"name": "get_weather"' in text
