import textwrap
import unittest
from datetime import datetime

from llama_toolchain.common.prompt_templates.system_prompts import (
    BuiltinToolGenerator,
    FunctionTagCustomToolGenerator,
    JsonCustomToolGenerator,
    SystemDefaultGenerator,
)


class PromptTemplateTests(unittest.TestCase):

    def check_generator_output(self, generator, expected_text):
        example = generator.data_examples()[0]

        pt = generator.gen(example)
        text = pt.render()
        # print(text)  # debugging
        self.assertEqual(text, expected_text)

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
            If a you choose to call a function ONLY reply in the following format with no prefix or suffix:

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
