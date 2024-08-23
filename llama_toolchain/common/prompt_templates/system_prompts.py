import textwrap
from datetime import datetime
from typing import Any, Dict, List

from llama_models.llama3.api.datatypes import (
    BuiltinTool,
    ToolDefinition,
    ToolParamDefinition,
)

from .base import PromptTemplate, PromptTemplateGeneratorBase


class SystemDefaultGenerator(PromptTemplateGeneratorBase):

    def gen(self, *args, **kwargs) -> PromptTemplate:
        template_str = textwrap.dedent(
            """
            Cutting Knowledge Date: December 2023
            Today Date: {{ today }}
            """
        )
        return PromptTemplate(
            template_str.lstrip("\n"),
            {"today": datetime.now().strftime("%d %B %Y")},
        )

    def data_examples(self) -> List[Any]:
        return [None]


class BuiltinToolGenerator(PromptTemplateGeneratorBase):

    def _tool_breakdown(self, tools: List[ToolDefinition]):
        builtin_tools, custom_tools = [], []
        for dfn in tools:
            if isinstance(dfn.tool_name, BuiltinTool):
                builtin_tools.append(dfn)
            else:
                custom_tools.append(dfn)

        return builtin_tools, custom_tools

    def gen(self, tools: List[ToolDefinition]) -> PromptTemplate:
        builtin_tools, custom_tools = self._tool_breakdown(tools)
        data = []
        template_str = textwrap.dedent(
            """
            {% if builtin_tools or custom_tools -%}
            Environment: ipython
            {% endif -%}
            {% set builtin_tools = builtin_tools | reject('equalto', 'code_interpreter') | list -%}
            {% if builtin_tools -%}
            Tools: {{ builtin_tools | join(", ") | trim -}}
            {% endif %}
            """
        )
        return PromptTemplate(
            template_str.lstrip("\n"),
            {
                "builtin_tools": [t.tool_name.value for t in builtin_tools],
                "custom_tools": custom_tools,
            },
        )

    def data_examples(self) -> List[List[ToolDefinition]]:
        return [
            # builtin tools
            [
                ToolDefinition(tool_name=BuiltinTool.code_interpreter),
                ToolDefinition(tool_name=BuiltinTool.brave_search),
                ToolDefinition(tool_name=BuiltinTool.wolfram_alpha),
            ],
            # only code interpretor
            [
                ToolDefinition(tool_name=BuiltinTool.code_interpreter),
            ],
        ]


class JsonCustomToolGenerator(PromptTemplateGeneratorBase):

    def gen(self, custom_tools: List[ToolDefinition]) -> PromptTemplate:
        template_str = textwrap.dedent(
            """
            Answer the user's question by making use of the following functions if needed.
            If none of the function can be used, please say so.
            Here is a list of functions in JSON format:
            {% for t in custom_tools -%}
            {# manually setting up JSON because jinja sorts keys in unexpected ways -#}
            {%- set tname = t.tool_name -%}
            {%- set tdesc = t.description -%}
            {%- set tparams = t.parameters -%}
            {%- set required_params = [] -%}
            {%- for name, param in tparams.items() if param.required == true -%}
                {%- set _ = required_params.append(name) -%}
            {%- endfor -%}
            {
                "type": "function",
                "function": {
                    "name": "{{tname}}",
                    "description": "{{tdesc}}",
                    "parameters": {
                        "type": "object",
                        "properties": [
                            {%- for name, param in tparams.items() %}
                            {
                                "{{name}}": {
                                    "type": "object",
                                    "description": "{{param.description}}"
                                }
                            }{% if not loop.last %},{% endif %}
                            {%- endfor %}
                        ],
                        "required": {{ required_params | tojson }}
                    }
                }
            }
            {% endfor %}
            Return function calls in JSON format.
            """
        )

        return PromptTemplate(
            template_str.lstrip("\n"),
            {"custom_tools": [t.model_dump() for t in custom_tools]},
        )

    def data_examples(self) -> List[List[ToolDefinition]]:
        return [
            [
                ToolDefinition(
                    tool_name="trending_songs",
                    description="Returns the trending songs on a Music site",
                    parameters={
                        "n": ToolParamDefinition(
                            param_type="int",
                            description="The number of songs to return",
                            required=True,
                        ),
                        "genre": ToolParamDefinition(
                            param_type="str",
                            description="The genre of the songs to return",
                            required=False,
                        ),
                    },
                ),
            ]
        ]


class FunctionTagCustomToolGenerator(PromptTemplateGeneratorBase):

    def gen(self, custom_tools: List[ToolDefinition]) -> PromptTemplate:
        template_str = textwrap.dedent(
            """
            You have access to the following functions:

            {% for t in custom_tools %}
            {#- manually setting up JSON because jinja sorts keys in unexpected ways -#}
            {%- set tname = t.tool_name -%}
            {%- set tdesc = t.description -%}
            {%- set tparams = t.parameters | tojson -%}
            Use the function '{{ tname }}' to '{{ tdesc }}':
            {"name": "{{tname}}", "description": "{{tdesc}}", "parameters": {{tparams}}}

            {% endfor -%}
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
        return PromptTemplate(
            template_str.lstrip("\n"),
            {"custom_tools": [t.model_dump() for t in custom_tools]},
        )

    def data_examples(self) -> List[List[ToolDefinition]]:
        return [
            [
                ToolDefinition(
                    tool_name="trending_songs",
                    description="Returns the trending songs on a Music site",
                    parameters={
                        "n": ToolParamDefinition(
                            param_type="int",
                            description="The number of songs to return",
                            required=True,
                        ),
                        "genre": ToolParamDefinition(
                            param_type="str",
                            description="The genre of the songs to return",
                            required=False,
                        ),
                    },
                ),
            ]
        ]
